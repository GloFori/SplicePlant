import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

# ---------------- DNA utils ----------------
DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}

def read_fasta_one(path: str) -> Tuple[str, str]:
    name = None
    seq_lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is None:
                    name = line[1:].split()[0]
                else:
                    break
            else:
                if name is not None:
                    seq_lines.append(line)
    if name is None:
        raise ValueError("Empty/invalid FASTA: %s" % path)
    return "".join(seq_lines), name

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
class ResBlock(nn.Module):
    def __init__(self, L, W, AR):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(L)
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding="same")
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding="same")

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        return out + x

try:
    from sinkhorn_transformer import SinkhornTransformer  # type: ignore
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
        self.W = np.asarray([11,11,11,11,11,11,11,11, 21,21,21,21,21,21,21,21])
        self.AR= np.asarray([ 1, 1, 1, 1, 4, 4, 4, 4, 10,10,10,10,20,20,20,20])
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip  = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, int(self.W[i]), int(self.AR[i])))
            if ((i+1) % 4 == 0) or ((i+1) == len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 40, 1)

    def forward(self, x):               # [B, L, 4]
        x = x.permute(0, 2, 1)          # [B, 4, L]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)
            if ((i+1) % 4 == 0) or ((i+1) == len(self.W)):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        out = self.splice_output(skip)  # [B, 40, L]
        return out

class StudentModel(nn.Module):
    def __init__(self, output_classes=3, dim=40, seq_len=401):
        super().__init__()
        self.encoder = SpEncoder(L=50)
        self.attn    = AttnBlock(dim=dim, pos_len=max(1200, seq_len))
        self.fc1     = nn.Linear(seq_len, 512)
        self.conv1   = nn.Conv1d(dim, output_classes, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(512, 1)

    def forward(self, x):               # [B, L, 4]
        x = self.encoder(x)             # [B, 40, L]
        x = self.fc1(x)                 # [B, 40, 512]
        x = self.relu(x)
        x = self.attn(x)                # [B, 40, 512]
        x = self.conv1(x)               # [B, 3, 512]
        x = self.dropout(x)
        x = self.fc2(x)                 # [B, 3, 1]
        x = torch.mean(x, dim=2)        # [B, 3]
        return x


# ---------------- Scoring + Calling ----------------
def infer_win_from_ckpt(state: Dict) -> Optional[int]:
    for k in ["fc1.weight", "module.fc1.weight"]:
        if k in state and hasattr(state[k], "shape"):
            return int(state[k].shape[1])
    return None

def load_state_dict_any(ckpt_path: str, device: torch.device) -> Dict:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint is not a state_dict or {'state_dict':...}")
    return state

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
    ap.add_argument("--fasta", required=True, help="Input FASTA (first record only).")
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

    state = load_state_dict_any(args.ckpt, device)
    win = args.win
    if win == 0:
        win = infer_win_from_ckpt(state)
        if win is None:
            raise RuntimeError("Cannot infer --win from checkpoint. Please set --win manually.")
    if win % 2 == 0:
        raise RuntimeError("--win must be odd, got %d" % win)

    model = StudentModel(output_classes=3, dim=40, seq_len=win).to(device)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        if (not SINKHORN_AVAILABLE) and ("Missing key(s)" in str(e) or "Unexpected key(s)" in str(e)):
            raise RuntimeError(
                "Checkpoint likely trained with sinkhorn_transformer, but it's not available here.\n"
                "Install the same dependency or run in the training environment.\n\n" + str(e)
            )
        raise

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
        p_for_calling = np.nan_to_num(p_mut, nan=0.0)

    label_suffix = "mut" if args.edit else "wt"
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
