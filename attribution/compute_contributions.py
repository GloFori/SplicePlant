# === compute_contributions.py ===
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from captum.attr import DeepLift
from collections import defaultdict
import argparse
import glob

def _maybe_import_modisco(enable: bool):
    if not enable:
        return None
    try:
        import modisco
        import modisco.backend
        import modisco.tfmodisco_workflow
        return modisco
    except Exception as e:
        raise RuntimeError(
            "TF-MoDISco is activated，but cannot import modisco \n"
            "please try --no_tfmodisco \n"
            f"{repr(e)}"
        )

class ResBlock(nn.Module):
    def __init__(self, L, W, AR):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(L)
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding='same')
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding='same')

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        return out + x

class AttnBlock(nn.Module):
    def __init__(self, dim=40, pos_len=1200):
        super().__init__()
        self.pos_emb = nn.Embedding(pos_len, dim)
        self.attn = nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, dim, Lp = x.size()
        pos_idx = torch.arange(0, Lp, device=x.device).unsqueeze(0).expand(b, Lp)
        x = x.transpose(1, 2).contiguous()
        x = x + self.pos_emb(pos_idx)
        x = self.attn(x)
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        return x

class SpEncoder(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21])
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20])
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 40, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                dense = self.convs[j](conv)
                skip = skip + dense
                j += 1
        out = self.splice_output(skip)
        return out

class StudentModel(nn.Module):
    def __init__(self, output_classes=3, dim=40, seq_len=401):
        super().__init__()
        self.encoder = SpEncoder(L=50)
        self.attn = AttnBlock(dim=dim, pos_len=max(1200, seq_len))
        self.fc1 = nn.Linear(seq_len, 512)
        self.conv1 = nn.Conv1d(dim, output_classes, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.attn(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.mean(x, dim=2)
        return x


_BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def seq_to_onehot(seq: str):
    seq = seq.upper()
    L = len(seq)
    x = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = _BASE2IDX.get(b, None)
        if j is not None:
            x[i, j] = 1.0
    return x

def build_markov_transitions(sequence):
    seq = sequence.upper()
    bases = ["A", "C", "G", "T"]
    counts = {b: {bb: 1.0 for bb in bases} for b in bases}  # Laplace
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a in counts and b in counts[a]:
            counts[a][b] += 1.0
    trans = {}
    for a in bases:
        s = sum(counts[a].values())
        trans[a] = {b: counts[a][b] / s for b in bases}
    return trans

def generate_markov_sequence(length, trans_probs):
    bases = ["A", "C", "G", "T"]
    seq = [np.random.choice(bases)]
    for _ in range(length - 1):
        last = seq[-1]
        p = np.array([trans_probs.get(last, {}).get(b, 0.25) for b in bases], dtype=float)
        p = p / p.sum()
        seq.append(np.random.choice(bases, p=p))
    return ''.join(seq)

# =======================
# batch DeepLIFT
# =======================
def _set_fast_cuda_flags():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

def compute_batch_contributions(
    model,
    window_seqs,
    device,
    trans_model,
    target_class: int,
    num_refs: int,
):
    B = len(window_seqs)
    L = len(window_seqs[0])
    x_np = np.stack([seq_to_onehot(s) for s in window_seqs], axis=0)  # [B,L,4]
    x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)

    dl = DeepLift(model)

    # hypothetical templates
    hypo_templates = torch.zeros((4, 1, L, 4), dtype=torch.float32, device=device)
    for b in range(4):
        hypo_templates[b, :, :, b] = 1.0

    real_acc = torch.zeros((B, L, 4), dtype=torch.float32, device=device)
    hyp_acc = torch.zeros((B, L, 4), dtype=torch.float32, device=device)
    score_acc = torch.zeros((B, L), dtype=torch.float32, device=device)

    for _ in range(num_refs):
        # batch baseline
        baseline_seqs = [generate_markov_sequence(len(s), trans_model) for s in window_seqs]
        baseline_np = np.stack([seq_to_onehot(s0) for s0 in baseline_seqs], axis=0)  # [B,L,4]
        baseline = torch.tensor(baseline_np, dtype=torch.float32, device=device)

        # attr: [B,L,4]
        attr = dl.attribute(x, baselines=baseline, target=int(target_class))

        real = attr * x                         # [B,L,4]
        score = real.sum(dim=-1)                # [B,L]
        diff = hypo_templates - baseline.unsqueeze(0)   # [4,B,L,4]
        hyp = (attr.unsqueeze(0) * diff).sum(dim=-1)    # [4,B,L]
        hyp = hyp.permute(1, 2, 0)                      # [B,L,4]

        real_acc += real
        hyp_acc += hyp
        score_acc += score

    real_acc /= float(num_refs)
    hyp_acc /= float(num_refs)
    score_acc /= float(num_refs)

    return (
        score_acc.detach().cpu().numpy(),   # [B,L]
        real_acc.detach().cpu().numpy(),    # [B,L,4]
        hyp_acc.detach().cpu().numpy(),     # [B,L,4]
    )

def save_motifs(motifs, output_file):
    base_order = ["A", "C", "G", "T"]
    with open(output_file, "w") as f:
        for k, m in enumerate(motifs, start=1):
            start0, end0 = int(m["start0"]), int(m["end0"])
            start1, end1 = start0 + 1, end0 + 1
            seq = m["seq"]
            scores = m["scores"]
            hyp = m["hyp"]
            Lmotif = len(seq)

            f.write(f">motif{k}\tpos0:{start0}-{end0}\tpos1:{start1}-{end1}\tlen:{Lmotif}\n")
            f.write(seq + "\n")
            f.write(" ".join(f"{s:.4f}" for s in scores) + "\n")
            f.write("#hypothetical_contrib (columns: A C G T)\n")
            f.write("global_pos0\tbase\tA\tC\tG\tT\n")
            for j in range(Lmotif):
                gpos0 = start0 + j
                b = seq[j]
                row = hyp[j, :]
                f.write(
                    f"{gpos0}\t{b}\t" +
                    "\t".join(f"{row[bi]:.6f}" for bi in range(4)) +
                    "\n"
                )
            f.write("\n")

def extract_motifs_from_windows_fast(
    model,
    sequence,
    labels,
    device,
    window_size=401,
    threshold=0.12,
    min_length=5,
    step=1,
    num_refs=3,
    collect_tfmodisco=False,
    batch_size=512,
):
    seq_len = len(sequence)
    side = window_size // 2

    centers = np.where((labels == 0) | (labels == 1))[0]
    starts = centers - side

    if step > 1:
        starts = np.array([s for s in starts.tolist() if (s % step) == 0], dtype=int)

    valid = (starts >= 0) & (starts + window_size <= seq_len)
    starts = starts[valid]
    if starts.size == 0:
        return [], defaultdict(list)

    tfmodisco_inputs = defaultdict(list)

    motifs = []

    def process_bucket(bucket_starts, target_class):
        for i in range(0, len(bucket_starts), batch_size):
            bs = bucket_starts[i:i + batch_size]
            window_seqs = [sequence[s:s + window_size] for s in bs]

            scores_B, real_B, hyp_B = compute_batch_contributions(
                model=model,
                window_seqs=window_seqs,
                device=device,
                trans_model=build_markov_transitions(sequence),
                target_class=int(target_class),
                num_refs=num_refs,
            )

            for j, start in enumerate(bs):
                window_seq = window_seqs[j]
                scores = scores_B[j]      # [L]
                hyp = hyp_B[j]            # [L,4]

                if collect_tfmodisco and np.max(np.abs(scores)) >= threshold:
                    onehot = seq_to_onehot(window_seq)
                    tfmodisco_inputs[int(target_class)].append((onehot, real_B[j], hyp_B[j]))

                current_start_i = None
                current_seq = []
                current_scores = []
                current_hyp = []

                for p in range(window_size):
                    if abs(scores[p]) >= threshold:
                        if current_start_i is None:
                            current_start_i = p
                        current_seq.append(window_seq[p])
                        current_scores.append(float(scores[p]))
                        current_hyp.append(hyp[p].copy())
                    else:
                        if current_start_i is not None:
                            if len(current_seq) >= min_length:
                                global_start0 = start + current_start_i
                                global_end0 = start + (p - 1)
                                motifs.append({
                                    "start0": global_start0,
                                    "end0": global_end0,
                                    "seq": "".join(current_seq),
                                    "scores": current_scores,
                                    "hyp": np.vstack(current_hyp),
                                })
                            current_start_i = None
                            current_seq = []
                            current_scores = []
                            current_hyp = []

                if current_start_i is not None and len(current_seq) >= min_length:
                    global_start0 = start + current_start_i
                    global_end0 = start + (window_size - 1)
                    motifs.append({
                        "start0": global_start0,
                        "end0": global_end0,
                        "seq": "".join(current_seq),
                        "scores": current_scores,
                        "hyp": np.vstack(current_hyp),
                    })

    # center = start + side
    target0_starts = []
    target1_starts = []
    for s in starts.tolist():
        c = s + side
        if labels[c] == 0:
            target0_starts.append(s)
        elif labels[c] == 1:
            target1_starts.append(s)

    if target0_starts:
        process_bucket(target0_starts, 0)
    if target1_starts:
        process_bucket(target1_starts, 1)

    return motifs, tfmodisco_inputs

def read_two_line_file(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    if len(lines) < 2:
        return None, None
    seq = lines[0].strip().upper()
    lab = lines[1].strip()
    if (" " in lab) or ("\t" in lab):
        labels = np.array([int(x) for x in lab.split()], dtype=int)
    else:
        labels = np.array([int(ch) for ch in lab], dtype=int)
    return seq, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--window_size", default=401, type=int)
    parser.add_argument("--threshold", default=0.12, type=float)
    parser.add_argument("--min_length", default=5, type=int)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--num_refs", default=3, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--no_tfmodisco", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_fast_cuda_flags()

    model = StudentModel(seq_len=args.window_size).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    collect_tfmodisco = (not args.no_tfmodisco)
    modisco = _maybe_import_modisco(collect_tfmodisco)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    for path in tqdm(files, desc="Processing"):
        seq, labels = read_two_line_file(path)
        if seq is None:
            continue
        if len(seq) != len(labels):
            print(f"[WARN] length mismatch: {path}", file=sys.stderr)
            continue

        motifs, tf_inputs = extract_motifs_from_windows_fast(
            model=model,
            sequence=seq,
            labels=labels,
            device=device,
            window_size=args.window_size,
            threshold=args.threshold,
            min_length=args.min_length,
            step=args.step,
            num_refs=args.num_refs,
            collect_tfmodisco=collect_tfmodisco,
            batch_size=args.batch_size,
        )

        base = os.path.splitext(os.path.basename(path))[0]
        out_motif = os.path.join(args.output_dir, f"{base}_motifs.txt")
        save_motifs(motifs, out_motif)

if __name__ == "__main__":
    main()
