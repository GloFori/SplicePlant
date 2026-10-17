"""Prepare new demo windows; archived manuscript inputs are distributed on Zenodo."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, help="Genome FASTA")
    parser.add_argument("--gff", required=True, help="Matching GFF3 with mRNA/exon features")
    parser.add_argument("--out", required=True, help="New output CSV")
    parser.add_argument("--flank", type=int, default=200, help="200 gives 401-bp windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=10000000)
    args = parser.parse_args()
    if args.flank < 1 or args.max_attempts < 1:
        parser.error("flank and max-attempts must be positive")
    from Bio import SeqIO
    import pandas as pd, re, tqdm, random
    random.seed(args.seed)
    FA, GFF, WINDOW = args.fasta, args.gff, args.flank
    genome = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(FA, "fasta")}

    tx_exons = {}   # {tx_id: {"chrom":..., "strand":..., "exons":[(start,end), ...]}}
    with open(GFF) as f:
        for line in f:
            if line.startswith("##FASTA"): break
            if not line.strip() or line.startswith("#"): continue
            chrom, source, feature, start, end, score, strand, phase, attr = line.strip().split("\t")
            start, end = int(start), int(end)

            if feature in ("mRNA", "transcript"):
                m = re.search(r"ID=([^;]+)", attr)
                if not m: continue
                tx_id = m.group(1)
                tx_exons.setdefault(tx_id, {"chrom": chrom, "strand": strand, "exons": []})

            elif feature == "exon":
                pm = re.search(r"Parent=([^;]+)", attr)
                if not pm: continue
                tx_id = pm.group(1)
                if tx_id not in tx_exons:
                    tx_exons[tx_id] = {"chrom": chrom, "strand": strand, "exons": []}
                tx_exons[tx_id]["exons"].append((start, end))

    def rc(dna: str) -> str:
        tb = str.maketrans("ACGTN", "TGCAN")
        return dna.translate(tb)[::-1]

    def get_window(chrom, center, strand, win=WINDOW):
        seq = genome.get(chrom)
        if seq is None: return None

        center_idx = center - 1
        left = center_idx - win
        right = center_idx + win
        if left < 0 or right >= len(seq): return None
        s = seq[left:right + 1]

        return rc(s) if strand == "-" else s

    donor_seqs, acceptor_seqs = [], []

    for tx_id, info in tqdm.tqdm(list(tx_exons.items())):
        chrom, strand, exons = info["chrom"], info["strand"], info["exons"]
        if not exons or chrom not in genome: continue

        exons.sort(key=lambda x: x[0])

        for i in range(len(exons) - 1):
            s1, e1 = exons[i]
            s2, e2 = exons[i + 1]

            if e1 + 1 >= s2 - 1: continue

            if strand == "+":
                donor_center = e1
                acceptor_center = s2
            else:
                donor_center = s2
                acceptor_center = e1

            dseq = get_window(chrom, donor_center, strand)
            aseq = get_window(chrom, acceptor_center, strand)

            if dseq: donor_seqs.append(dseq)
            if aseq: acceptor_seqs.append(aseq)

    donor_df = pd.DataFrame({"seq": donor_seqs, "label": 1})
    acceptor_df = pd.DataFrame({"seq": acceptor_seqs, "label": 0})

    total_pos_len = len(donor_df) + len(acceptor_df)
    print("Donors (label=1):", len(donor_df))
    print("Acceptors (label=0):", len(acceptor_df))
    print("Total Positives:", total_pos_len)

    used = set(donor_df["seq"]) | set(acceptor_df["seq"])
    neg = []
    all_chroms = list(genome.keys())
    target_neg = total_pos_len

    eligible = [chrom for chrom in all_chroms if len(genome[chrom]) >= 2*WINDOW+5]
    if not eligible or not total_pos_len:
        raise ValueError("No usable positive windows or chromosomes; check FASTA/GFF3 names and flank")
    attempts = 0
    while len(neg) < target_neg:
        attempts += 1
        if attempts > args.max_attempts:
            raise RuntimeError("Negative sampling limit reached; check genome content or increase --max-attempts")
        chrom = random.choice(eligible)
        seq = genome[chrom]
        if len(seq) < 2*WINDOW+5: continue
        c = random.randint(WINDOW+1, len(seq)-WINDOW-2)
        w = seq[c-WINDOW:c+WINDOW+1]
        if "N" in w: continue
        if w in used: continue
        neg.append(w)

    neg_df = pd.DataFrame({"seq": neg, "label": 2})
    print("Neither (label=2):", len(neg_df))

    df = pd.concat([donor_df, acceptor_df, neg_df])

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    output_filename = args.out
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    if Path(output_filename).exists():
        raise FileExistsError(output_filename)
    df.to_csv(output_filename, index=False)

    print(f"\nSaved full dataset to {output_filename}")
    print("Full dataset label distribution:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
