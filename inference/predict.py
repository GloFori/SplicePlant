import os
import argparse
import sys, argparse, time, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

log_file = None
def init_log(run_name, log_dir):
    global log_file
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{run_name}_trainlog_{now}.txt")
    log_file = open(path, "w")
    return path

def log(msg):
    print(msg)
    if log_file:
        log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        log_file.flush()

# -----------------------------
# DNA utilities
# -----------------------------
DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode(seq: str) -> np.ndarray:
    L = len(seq)
    x = np.zeros((L, 4), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in DNA_VOCAB:
            x[i, DNA_VOCAB[ch]] = 1.0
    return x

# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = one_hot_encode(self.seqs[idx])  # [L,4]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# -----------------------------
# Student model
# -----------------------------
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, L, W, AR):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(L)
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding='same')
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding='same')
    def forward(self, x):
        out = self.bn1(x); out = torch.relu(out); out = self.conv1(out)
        out = self.bn2(out); out = torch.relu(out); out = self.conv2(out)
        return out + x

# Optional Sinkhorn (if available)
try:
    from sinkhorn_transformer import SinkhornTransformer
    SINKHORN_AVAILABLE = True
except Exception:
    SINKHORN_AVAILABLE = False

class AttnBlock(nn.Module):
    def __init__(self, dim=40, pos_len=1200, depth=6, causal=False, reversible=False):
        super().__init__()
        self.pos_emb = nn.Embedding(pos_len, dim)
        if SINKHORN_AVAILABLE:
            self.attn = SinkhornTransformer(
                dim, depth,
                heads=8, n_local_attn_heads=2,
                attn_layer_dropout=0.1, layer_dropout=0.1,
                ff_dropout=0.1, ff_chunks=10,
                causal=causal, reversible=reversible, non_permutative=True
            )
        else:
            self.attn = nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # [B, dim, L']
        b, dim, Lp = x.size()
        pos_idx = torch.arange(0, Lp, device=x.device).unsqueeze(0).expand(b, Lp)
        x = x.transpose(1, 2).contiguous()  # [B, L', dim]
        x = x + self.pos_emb(pos_idx)
        x = self.attn(x)
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        return x

class SpEncoder(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        self.W  = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21])
        self.AR = np.asarray([ 1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20])
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip  = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i+1)%4==0) or ((i+1)==len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 40, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)
            if ((i+1)%4==0) or ((i+1)==len(self.W)):
                dense = self.convs[j](conv); j += 1
                skip = skip + dense
        out = self.splice_output(skip)  # [B, 40, L]
        return out

class StudentModel(nn.Module):
    def __init__(self, output_classes=3, dim=40, seq_len=1001):
        super().__init__()
        self.encoder = SpEncoder(L=50)                  # -> [B, 40, L]
        self.attn = AttnBlock(dim=dim, pos_len=max(1200, seq_len))
        self.fc1 = nn.Linear(seq_len, 512)          # per-channel projection
        self.conv1 = nn.Conv1d(dim, output_classes, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):               # x: [B, L, 4]
        x = self.encoder(x)             # [B, 40, L]
        x = self.fc1(x)                 # [B, 40, 512]
        x = self.relu(x)
        x = self.attn(x)                # [B, 40, 512]
        x = self.conv1(x)               # [B, 3, 512]
        x = self.dropout(x)
        x = self.fc2(x)                 # [B, 3, 1]
        x = torch.mean(x, dim=2)        # [B, 3]
        return x

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_prob.append(probs)
            all_pred.append(preds)
            all_y.append(y.numpy())
    all_prob = np.concatenate(all_prob, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    acc = accuracy_score(all_y, all_pred)
    auc_macro = None
    try:
        auc_macro = roc_auc_score(all_y, all_prob, multi_class='ovr', average='macro')
    except ValueError:
        pass
    return acc, auc_macro, all_y, all_pred, all_prob

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    ap.add_argument("--weights", type=str, default="./save_model/student_best.pth", help="Path to trained Student weights")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_dir", type=str, default="./logs")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    assert "seq" in df.columns and "label" in df.columns, "CSV must have 'seq' and 'label' columns"
    seqs = df["seq"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    # Build dataset / loader
    ds = SeqDataset(seqs, labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Build model & load weights
    seq_len = len(seqs[0])
    device = torch.device(args.device)
    model = StudentModel(output_classes=3, dim=40, seq_len=seq_len).to(device)

    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)

    # Evaluate
    acc, auc_macro, y_true, y_pred, y_prob = evaluate(model, loader, device)

    log_path = init_log("predict", args.log_dir)
    log(f"Log file: {log_path}")
    log(f"use the weights of {args.weights}, test in data {args.csv}")
    log(f"\n[RESULT] Accuracy: {acc:.4f}")
    if auc_macro is not None:
        log(f"[RESULT] Macro AUC (OVR): {auc_macro:.4f}")
    else:
        log("[RESULT] Macro AUC (OVR): N/A (one or more classes missing in ground-truth)")

    log("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=["Acceptor (0)", "Donor (1)", "Neither (2)"], digits=4)
    log(report)
    if log_file: log_file.close()

if __name__ == "__main__":
    main()