import os
import argparse
import sys, argparse, time, datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spliceplant.model import StudentModel, load_student, read_checkpoint, checkpoint_window
from spliceplant.inputs import read_fasta, read_labeled_csv

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, roc_auc_score
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
    from sklearn.metrics import classification_report

    # Load data
    seqs, labels_list = read_labeled_csv(args.csv)
    labels = np.asarray(labels_list, dtype=np.int64)
    if args.batch_size < 1:
        ap.error("batch_size must be positive")

    # Build dataset / loader
    ds = SeqDataset(seqs, labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Build model & load weights
    seq_len = len(seqs[0])
    device = torch.device(args.device)
    model, _ = load_student(args.weights, device, window=seq_len)

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
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], zero_division=0, target_names=["Acceptor (0)", "Donor (1)", "Neither (2)"], digits=4)
    log(report)
    if log_file: log_file.close()

if __name__ == "__main__":
    main()
