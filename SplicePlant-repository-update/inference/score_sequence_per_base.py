import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

# ---------------- DNA utils ----------------
DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}

def read_fasta_one(path):
    records = read_fasta(path)
    if len(records) != 1:
        raise ValueError("Provide exactly one FASTA record; this scorer does not silently discard later records.")
    name, sequence = records[0]
    return sequence, name.split()[0]

def one_hot_encode(seq: str) -> np.ndarray:
    s = seq.upper()
    x = np.zeros((len(s), 4), dtype=np.float32)
    for i, ch in enumerate(s):
        j = DNA_VOCAB.get(ch, None)
        if j is not None:
            x[i, j] = 1.0
    return x

def apply_single_edit(seq: str, edit: str) -> Tuple[str, Dict]:
    # format: 201:G>A  or 201:A
    m = re.match(r"^(\d+):([ACGTNacgtn])(?:>([ACGTNacgtn]))?$", edit.strip())
    if not m:
        raise ValueError('Bad --edit. Use "201:G>A" or "201:A"')
    pos1 = int(m.group(1))
    ref = m.group(2).upper()
    alt = m.group(3).upper() if m.group(3) is not None else ref

    if pos1 < 1 or pos1 > len(seq):
        raise ValueError("Edit position out of range: %d (len=%d)" % (pos1, len(seq)))

    i = pos1 - 1
    wt = seq[i].upper()
    if m.group(3) is not None and wt != ref:
        raise ValueError(f"Reference mismatch at {pos1}: expected {ref}, found {wt}")
    mut = list(seq)
    mut[i] = alt
    return "".join(mut), {"pos1": pos1, "wt_base": wt, "ref": ref, "alt": alt}

def apply_motif_deletion(seq: str, spec: str) -> Tuple[str, Dict]:
    """
    spec format: "201:TAGAA" (1-based start position + motif string)
    Performs a true deletion in the mutant sequence (length shortens),
    but meta carries enough info to align predictions back to WT coordinates.
    """
    m = re.match(r"^(\d+):([ACGTNacgtn]+)$", spec.strip())
    if not m:
        raise ValueError('Bad --del. Use "201:TAGAA" (1-based start + motif string)')

    start1 = int(m.group(1))
    motif = m.group(2).upper()
    if start1 < 1 or start1 > len(seq):
        raise ValueError(f"Deletion start out of range: {start1} (len={len(seq)})")

    start0 = start1 - 1
    end0 = start0 + len(motif)  # python slicing end (exclusive)
    if end0 > len(seq):
        raise ValueError(f"Deletion exceeds sequence length: start={start1}, motif_len={len(motif)}, len={len(seq)}")

    wt_segment = seq[start0:end0].upper()
    if wt_segment != motif:
        raise ValueError(
            f'WT segment mismatch at {start1}: expected "{motif}", got "{wt_segment}". '
            f'If you want to force deletion anyway, relax this check.'
        )

    mut_seq = seq[:start0] + seq[end0:]  # true deletion

    meta = {
        "type": "DEL",
        "start1": start1,
        "end1": start1 + len(motif) - 1,
        "motif": motif,
        "del_len": len(motif),
        "start0": start0,
        "edit_str": f"DEL@{start1}:{motif}",
    }
    return mut_seq, meta


def align_probs_after_deletion(p_mut_raw: np.ndarray, L_wt: int, start0: int, del_len: int) -> np.ndarray:
    """
    Align mutant probabilities (computed on the shortened sequence) back to WT length.
    - Positions in [start0, start0+del_len) become NaN (gap).
    - Positions after the deletion are shifted: WT i maps to mut (i - del_len).
    """
    p_mut_aligned = np.full((L_wt, 3), np.nan, dtype=np.float32)

    # before deletion: 0..start0-1 unchanged
    if start0 > 0:
        p_mut_aligned[:start0, :] = p_mut_raw[:start0, :]

    # after deletion: WT index i >= start0+del_len maps to mut index (i - del_len)
    after0 = start0 + del_len
    if after0 < L_wt:
        src_start = start0
        src_end = p_mut_raw.shape[0]
        dst_start = after0
        dst_end = min(L_wt, dst_start + (src_end - src_start))
        p_mut_aligned[dst_start:dst_end, :] = p_mut_raw[src_start:src_start + (dst_end - dst_start), :]

    return p_mut_aligned

# ---------------- Model ----------------
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spliceplant.model import StudentModel, load_student, read_checkpoint, checkpoint_window
from spliceplant.inputs import read_fasta, read_labeled_csv

# ---------------- Scoring + Calling ----------------
def infer_win_from_ckpt(state: Dict) -> Optional[int]:
    for k in ["fc1.weight", "module.fc1.weight"]:
        if k in state and hasattr(state[k], "shape"):
            return int(state[k].shape[1])
    return None

def load_state_dict_any(ckpt_path, device=None):
    return read_checkpoint(ckpt_path)

def score_per_base(seq: str, model: nn.Module, win: int, batch_size: int, device: torch.device) -> np.ndarray:
    if win % 2 == 0:
        raise ValueError("--win must be odd.")
    L = len(seq)
    oh = one_hot_encode(seq)
    pad = win // 2
    padded = np.zeros((L + 2*pad, 4), dtype=np.float32)
    padded[pad:pad+L, :] = oh

    probs = np.zeros((L, 3), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, L, batch_size):
            e = min(L, s + batch_size)
            xs = np.stack([padded[i:i+win, :] for i in range(s, e)], axis=0)  # [B,win,4]
            xb = torch.from_numpy(xs).to(device=device, dtype=torch.float32)
            logits = model(xb)
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs[s:e, :] = p[:, :3]
    return probs

def call_peaks(probs: np.ndarray,
               cls: int,
               min_prob: float,
               min_margin: float,
               nms_radius: int,
               topn: int) -> List[Tuple[int, float, float]]:
    """
    probs: [L,3]
    cls: 0 acceptor, 1 donor
    Returns list of (pos1, score, margin) after thresholding + NMS, sorted by score desc.
    """
    L = probs.shape[0]
    score = probs[:, cls]
    other = np.max(np.delete(probs, cls, axis=1), axis=1)
    margin = score - other

    cand = []
    for i in range(L):
        if score[i] >= min_prob and margin[i] >= min_margin:
            cand.append((i+1, float(score[i]), float(margin[i])))

    cand.sort(key=lambda x: x[1], reverse=True)

    picked = []
    used = np.zeros(L+2, dtype=np.uint8)
    for pos1, sc, mg in cand:
        if used[pos1]:
            continue
        picked.append((pos1, sc, mg))
        # NMS suppress +/- radius
        lo = max(1, pos1 - nms_radius)
        hi = min(L, pos1 + nms_radius)
        used[lo:hi+1] = 1
        if len(picked) >= topn:
            break
    return picked

def write_scores_tsv(out_path: str, name: str, seq: str,
                     p_wt: np.ndarray,
                     edit_meta: Optional[Dict] = None,
                     p_mut: Optional[np.ndarray] = None) -> None:
    bases = list(seq.upper())
    with open(out_path, "w") as f:
        if p_mut is None:
            f.write("name\tpos0\tpos1\tbase\tpA\tpD\tpN\n")
            for i in range(len(seq)):
                f.write("%s\t%d\t%d\t%s\t%.6f\t%.6f\t%.6f\n" %
                        (name, i, i+1, bases[i], p_wt[i,0], p_wt[i,1], p_wt[i,2]))
        else:
            if edit_meta is not None and "edit_str" in edit_meta:
                edit_str = str(edit_meta["edit_str"])
            else:
                edit_str = "%d:%s>%s" % (edit_meta["pos1"], edit_meta["wt_base"], edit_meta["alt"])
            f.write("name\tpos0\tpos1\tbase\tpA_wt\tpD_wt\tpN_wt\tpA_mut\tpD_mut\tpN_mut\tdA\tdD\tdN\tedit\n")
            for i in range(len(seq)):
                d = p_mut[i] - p_wt[i]
                f.write("%s\t%d\t%d\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%s\n" %
                        (name, i, i+1, bases[i],
                         p_wt[i,0], p_wt[i,1], p_wt[i,2],
                         p_mut[i,0], p_mut[i,1], p_mut[i,2],
                         d[0], d[1], d[2], edit_str))

def write_calls_tsv(out_path: str, name: str, calls: List[Tuple[int, float, float]], label: str) -> None:
    with open(out_path, "w") as f:
        f.write("name\tpos1\tscore\tmargin\tlabel\n")
        for pos1, sc, mg in calls:
            f.write("%s\t%d\t%.6f\t%.6f\t%s\n" % (name, pos1, sc, mg, label))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="Input FASTA with exactly one record, in the orientation to be scored.")
    ap.add_argument("--ckpt", required=True, help="Model weights .pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)

    ap.add_argument("--win", type=int, default=0,
                    help="Window length. If 0, infer from checkpoint fc1.weight.")
    ap.add_argument("--out", default="scores.tsv", help="Output per-base TSV.")

    # mutation specs (mutually exclusive)
    mg = ap.add_mutually_exclusive_group()
    mg.add_argument("--edit", default=None, help='Optional SNP like "201:G>A". Put in quotes!')
    mg.add_argument("--del", dest="del_motif", default=None,
                    help='Optional motif deletion like "201:TAGAA" (1-based start + motif)')

    # calling params
    ap.add_argument("--min-prob", type=float, default=0.8, help="Min prob for calling a site.")
    ap.add_argument("--min-margin", type=float, default=0.2, help="Min margin: p_cls - max(p_other).")
    ap.add_argument("--nms-radius", type=int, default=10, help="NMS radius (bp).")
    ap.add_argument("--topn", type=int, default=50, help="Max calls per class.")
    ap.add_argument("--call-prefix", default="calls", help="Prefix for called-sites TSVs.")

    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    seq, name = read_fasta_one(args.fasta)

    if args.batch_size < 1 or args.nms_radius < 0 or args.topn < 1:
        raise ValueError("batch-size/topn must be positive; nms-radius must be nonnegative")
    if args.win < 0 or not 0 <= args.min_prob <= 1 or not -1 <= args.min_margin <= 1:
        raise ValueError("Invalid window or probability thresholds")
    model, win = load_student(args.ckpt, device, window=args.win)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.call_prefix).parent.mkdir(parents=True, exist_ok=True)

    # WT
    p_wt = score_per_base(seq, model, win=win, batch_size=args.batch_size, device=device)

    p_mut = None
    meta = None

    if args.edit:
        mut_seq, meta = apply_single_edit(seq, args.edit)
        p_mut_raw = score_per_base(mut_seq, model, win=win, batch_size=args.batch_size, device=device)
        p_mut = p_mut_raw

        write_scores_tsv(args.out, name, seq, p_wt, edit_meta=meta, p_mut=p_mut)
        print("[OK] wrote per-base WT+MUT+delta:", args.out, "edit=", meta.get("edit_str", args.edit))

    elif args.del_motif:
        mut_seq, meta = apply_motif_deletion(seq, args.del_motif)
        p_mut_raw = score_per_base(mut_seq, model, win=win, batch_size=args.batch_size, device=device)

        # align back to WT coords so output length/columns unchanged
        p_mut = align_probs_after_deletion(p_mut_raw, L_wt=len(seq), start0=meta["start0"], del_len=meta["del_len"])

        write_scores_tsv(args.out, name, seq, p_wt, edit_meta=meta, p_mut=p_mut)
        print("[OK] wrote per-base WT+MUT+delta:", args.out, "edit=", meta["edit_str"])

    else:
        write_scores_tsv(args.out, name, seq, p_wt)
        print("[OK] wrote per-base WT:", args.out)

    # Calling on WT only
    p_for_calling = p_wt
    if (args.edit or args.del_motif) and (p_mut is not None):
        p_for_calling = p_mut

    label_suffix = "mut" if (args.edit or args.del_motif) else "wt"
    print(f"Calling peaks on {label_suffix} sequence...")

    donor_calls = call_peaks(p_for_calling, cls=1,
                             min_prob=args.min_prob,
                             min_margin=args.min_margin,
                             nms_radius=args.nms_radius,
                             topn=args.topn)

    acc_calls = call_peaks(p_for_calling, cls=0,
                           min_prob=args.min_prob,
                           min_margin=args.min_margin,
                           nms_radius=args.nms_radius,
                           topn=args.topn)

    out_d = "%s.donor.tsv" % args.call_prefix
    out_a = "%s.acceptor.tsv" % args.call_prefix
    write_calls_tsv(out_d, name, donor_calls, "donor")
    write_calls_tsv(out_a, name, acc_calls, "acceptor")

    print("[OK] wrote called sites:", out_d, out_a)
    print("  (win=%d, min_prob=%.2f, min_margin=%.2f, nms_radius=%d)" %
          (win, args.min_prob, args.min_margin, args.nms_radius))


if __name__ == "__main__":
    main()

