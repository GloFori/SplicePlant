# kd_splice.py (patched)
import os, sys, argparse, time, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---- env
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_FUSER_DISABLE", "1")

# ---- logging
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

# ---- DNA utils
DNA_VOCAB = {"A":0,"C":1,"G":2,"T":3}
def one_hot_encode(seq):
    L = len(seq); x = np.zeros((L,4), dtype=np.float32)
    for i,ch in enumerate(seq.upper()):
        if ch in DNA_VOCAB: x[i, DNA_VOCAB[ch]] = 1.0
    return x

def kmer_tokenize(seq, k=3):
    s = seq.upper()
    return " ".join(s[i:i+k] for i in range(len(s)-k+1))

def soften_probs(p, tau=1.0):
    if tau == 1.0: return p
    p = np.power(np.clip(p, 1e-12, 1.0), 1.0/float(tau))
    p = p / p.sum(axis=1, keepdims=True)
    return p

# ---- dataset
class KDSpliceDataset(Dataset):
    def __init__(self, seqs, labels, teacher_probs, center_mask_prob=0.3):
        self.seqs = seqs
        self.labels = labels.astype(np.int64)
        self.teacher_probs = teacher_probs.astype(np.float32)
        self.center_mask_prob = center_mask_prob
        assert len(self.seqs) == len(self.labels) == len(self.teacher_probs)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        y = self.labels[idx]
        tp = self.teacher_probs[idx]

        x = one_hot_encode(seq)
        if np.random.rand() < self.center_mask_prob:
            center = x.shape[0] // 2
            x[center, :] = 0.0

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(tp, dtype=torch.float32)
        )

# ---- student
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

    def forward(self, x):
        b, dim, Lp = x.size()
        pos_idx = torch.arange(0, Lp, device=x.device).unsqueeze(0).expand(b, Lp)
        x = x.transpose(1,2).contiguous()  # [B, L', dim]
        x = x + self.pos_emb(pos_idx)
        x = self.attn(x)
        x = self.norm(x)
        x = x.transpose(1,2).contiguous()
        return x

class SpEncoder(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        self.W = np.asarray([11,11,11,11,11,11,11,11, 21,21,21,21,21,21,21,21])
        self.AR= np.asarray([ 1, 1, 1, 1, 4, 4, 4, 4, 10,10,10,10,20,20,20,20])
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip  = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i+1)%4==0) or ((i+1)==len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 40, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)
            if ((i+1)%4==0) or ((i+1)==len(self.W)):
                dense = self.convs[j](conv); j += 1
                skip = skip + dense
        out = self.splice_output(skip)
        return out

class StudentModel(nn.Module):
    def __init__(self, output_classes=3, dim=40, seq_len=1001):
        super().__init__()
        self.encoder = SpEncoder(L=50)
        self.attn = AttnBlock(dim=dim, pos_len=max(1200, seq_len))
        self.fc1 = nn.Linear(seq_len, 512)
        self.conv1 = nn.Conv1d(dim, output_classes, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):             # [B, L, 4]
        x = self.encoder(x)           # [B, 40, L]
        x = self.fc1(x)               # [B, 40, 512]
        x = self.relu(x)
        x = self.attn(x)              # [B, 40, 512]
        x = self.conv1(x)             # [B, 3, 512]
        x = self.dropout(x)
        x = self.fc2(x)               # [B, 3, 1]
        x = torch.mean(x, dim=2)      # [B, 3]
        return x

# ---- teacher: DNABERT-2
def build_dnabert(CKPT, device):
    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    for k in ["use_flash_attn", "flash_attn", "use_triton", "attn_impl"]:
        if hasattr(cfg, k):
            try:
                if k == "attn_impl":
                    setattr(cfg, k, "eager")
                else:
                    setattr(cfg, k, False)
            except Exception:
                pass
    cfg.return_dict = True
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    # enc = AutoModel.from_pretrained(CKPT, trust_remote_code=True, config=cfg).eval().to(device)
    enc = AutoModel.from_pretrained(CKPT, trust_remote_code=True, config=cfg).eval().to(
        "cuda" if torch.cuda.is_available() else "cpu")

    enc.config.return_dict = True
    enc.config.use_flash_attn = False
    return tok, enc

@torch.no_grad()
def mean_pool(last_hidden_state, attn_mask):
    mask = attn_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

def get_last_hidden(enc, **inputs):
    out = enc(**inputs, output_hidden_states=False, return_dict=True)
    if isinstance(out, tuple):
        return out[0]
    # ModelOutput
    return out.last_hidden_state

def encode_texts(texts, tok, enc, batch=32, max_len=512, device="cpu"):
    feats = []
    enc.eval()
    for i in tqdm(range(0, len(texts), batch), desc="Encoding with DNABERT-2"):
        batch_text = texts[i:i+batch]
        inputs = tok(batch_text, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            hidden = get_last_hidden(enc, **inputs)   # [B, L, H]
            vec = mean_pool(hidden, inputs["attention_mask"])  # [B, H]
        feats.append(vec.cpu().numpy())
    return np.vstack(feats)

# ---- eval
def evaluate(student, loader, device):
    student.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, _tp in loader:
            x, y = x.to(device), y.to(device)
            logits = student(x)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / max(1,total)

# ---- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data.csv", help="CSV with columns: seq,label")
    ap.add_argument("--ckpt", type=str, default="./dnabert2_117M")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.5, help="CE weight")
    ap.add_argument("--tau", type=float, default=1.0, help="temperature to soften teacher probs (>=1)")
    ap.add_argument("--center_mask_prob", type=float, default=0.3)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_dir", type=str, default="./logs")
    ap.add_argument("--save_path", type=str, default="./save_model/student_best.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load data
    df = pd.read_csv(args.csv)
    assert "seq" in df.columns and "label" in df.columns, "CSV must have seq,label columns"
    seqs = df["seq"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    # 2) split
    tr_idx, te_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=args.seed)
    seqs_tr = [seqs[i] for i in tr_idx]; y_tr = labels[tr_idx]
    seqs_te = [seqs[i] for i in te_idx]; y_te = labels[te_idx]

    # 3) teacher: encode + LR head
    tok, enc = build_dnabert(args.ckpt, device)
    texts_tr = [kmer_tokenize(s, k=3) for s in seqs_tr]
    texts_te = [kmer_tokenize(s, k=3) for s in seqs_te]

    Xtr = encode_texts(texts_tr, tok, enc, batch=32, max_len=args.max_len, device=device)
    Xte = encode_texts(texts_te, tok, enc, batch=32, max_len=args.max_len, device=device)

    clf = LogisticRegression(max_iter=500, n_jobs=-1, multi_class='multinomial')
    clf.fit(Xtr, y_tr)
    prob_tr = clf.predict_proba(Xtr)
    prob_te = clf.predict_proba(Xte)

    ytr_pred = clf.predict(Xtr)
    log(f"Teacher LR head (train) acc: {accuracy_score(y_tr, ytr_pred):.4f}")

    prob_tr = soften_probs(prob_tr, tau=args.tau)
    prob_te = soften_probs(prob_te, tau=args.tau)

    # 4) datasets
    train_ds = KDSpliceDataset(seqs_tr, y_tr, prob_tr, center_mask_prob=args.center_mask_prob)
    test_ds = KDSpliceDataset(seqs_te, y_te, prob_te, center_mask_prob=0.0)

    class_counts = np.bincount(y_tr)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # 5) student
    seq_len = len(seqs_tr[0])
    student = StudentModel(output_classes=3, dim=40, seq_len=seq_len).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    ce_crit = nn.CrossEntropyLoss()

    log_path = init_log("kd_splice", args.log_dir)
    log(f"Log file: {log_path}")
    log(f"Data: {args.csv}, Train={len(train_ds)}, Test={len(test_ds)}")
    log(f"alpha={args.alpha}, tau={args.tau}, lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}")

    best_val = -1.0
    patience, bad = 5, 0

    # 6) train loop (CE + KD)
    total_start_time = time.time()
    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        student.train()
        run_loss, tot, cor = 0.0, 0, 0
        for x, y, tp in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y, tp = x.to(device), y.to(device), tp.to(device)
            logits = student(x)
            ce = ce_crit(logits, y)
            kd = F.kl_div(F.log_softmax(logits, dim=1), tp, reduction='batchmean')  # KD
            loss = args.alpha * ce + (1.0 - args.alpha) * kd

            opt.zero_grad(); loss.backward(); opt.step()

            run_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            tot += y.size(0); cor += (pred == y).sum().item()

        epoch_duration = time.time() - epoch_start
        samples_per_sec = tot / epoch_duration
        mem_usage = ""
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            mem_usage = f"| GPU Mem: {max_mem:.2f} GB"
            torch.cuda.reset_peak_memory_stats(device)
        train_acc = cor / max(1,tot)
        val_acc   = evaluate(student, test_loader, device)
        log(f"Epoch {epoch:02d} | train_loss={run_loss/max(1,tot):.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        log_msg = (
            f"Epoch {epoch:02d} | "
            f"EpochTime(s): {epoch_duration:.2f} | "
            f"Throughput(samples/s): {samples_per_sec:.1f} | "
            f"TrainLoss: {run_loss / max(1, tot):.4f} | "
            f"TrainAcc: {train_acc:.4f} | "
            f"ValAcc: {val_acc:.4f} "
            f"{mem_usage}"
        )

        log(log_msg)

        if val_acc > best_val + 1e-4:
            best_val, bad = val_acc, 0
            torch.save(student.state_dict(), args.save_path)
            log(f"  ↑ New best. Saved to {args.save_path}")
        else:
            bad += 1
            if bad >= patience:
                log("Early stopping triggered.")
                break
    total_time = time.time() - total_start_time
    log(f"\n[Training Finished] Total Time: {str(datetime.timedelta(seconds=int(total_time)))}")

    # 7) final eval
    student.load_state_dict(torch.load(args.save_path, map_location=device))
    student.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for x, y, _tp in test_loader:
            x = x.to(device)
            logits = student(x)
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_y.append(y.numpy())
    all_pred = np.concatenate(all_pred); all_y = np.concatenate(all_y)
    log(f"[FINAL] Test Accuracy: {accuracy_score(all_y, all_pred):.4f}")
    log("\nClassification Report:\n" + classification_report(all_y, all_pred,
        target_names=["Acceptor (0)", "Donor (1)", "Neither (2)"]))

    if log_file: log_file.close()

if __name__ == "__main__":
    main()
