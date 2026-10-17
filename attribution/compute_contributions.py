# === compute_contributions.py ===
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
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

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from spliceplant_model import StudentModel, load_student, read_checkpoint, checkpoint_window
from spliceplant_inputs import read_fasta, read_labeled_csv

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

    from captum.attr import DeepLift
    dl = DeepLift(model)  # Legacy scoring formula retained; see attribution/README.md.

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
    if len(lines) != 2:
        raise ValueError(f"Expected exactly two lines in {path}; split multi-record files first.")
    seq = lines[0].strip().upper()
    lab = lines[1].strip()
    if (" " in lab) or ("\t" in lab):
        labels = np.array([int(x) for x in lab.split()], dtype=int)
    else:
        labels = np.array([int(ch) for ch in lab], dtype=int)
    if len(seq) != len(labels) or not set(labels.tolist()) <= {0, 1, 2}:
        raise ValueError(f"Invalid labels or sequence/label length mismatch: {path}")
    return seq, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--window_size", default=0, type=int, help="0: infer from checkpoint")
    parser.add_argument("--threshold", default=0.12, type=float)
    parser.add_argument("--min_length", default=5, type=int)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--num_refs", default=3, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--no_tfmodisco", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export_tfmodisco_inputs", action="store_true",
                        help="Export arrays only; no TF-MoDISco clustering is run")
    args = parser.parse_args()
    if min(args.step, args.num_refs, args.batch_size, args.min_length) < 1 or args.threshold < 0 or args.window_size < 0:
        parser.error("Counts must be positive; threshold/window_size must be nonnegative")
    if args.no_tfmodisco and args.export_tfmodisco_inputs:
        parser.error("--no_tfmodisco conflicts with --export_tfmodisco_inputs")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        _set_fast_cuda_flags()
    model, args.window_size = load_student(args.weights, device, args.window_size)
    collect_tfmodisco = args.export_tfmodisco_inputs
    print("Using archived attribution formula; see attribution/README.md for interpretation limits.")

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
        if collect_tfmodisco:
            for target, items in tf_inputs.items():
                np.savez_compressed(Path(args.output_dir) / f"{base}.class{target}.npz",
                    one_hot=np.stack([v[0] for v in items]),
                    contribution_scores=np.stack([v[1] for v in items]),
                    hypothetical_scores=np.stack([v[2] for v in items]))
    import json
    (Path(args.output_dir) / "run_config.json").write_text(json.dumps({"arguments": vars(args),
        "attribution_formula": "legacy_repository", "torch_version": torch.__version__,
        "tfmodisco_clustering_run": False}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

