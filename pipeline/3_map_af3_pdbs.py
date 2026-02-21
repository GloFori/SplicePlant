import os, sys, hashlib
from pathlib import Path
from collections import defaultdict

from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
FASTA = "/root/as/case/FLM_alt_pep.fa"  # input for the "cotton.py" program, is a FASTA file
PDB_DIR = "/root/as/case/af3_predictions"  # PDB output directory
OUT_DIR = "/root/as/case/af3_predictions_mapped"  # presenting the results (soft links) of the new directory

def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def seq_from_pdb(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    ppb = PPBuilder()
    chains = defaultdict(str)
    for model in structure:
        for chain in model:
            seqs = ppb.build_peptides(chain)
            if seqs:
                s = "".join(str(pp.get_sequence()) for pp in seqs)
                chains[chain.id] = s
        break
    return dict(chains)

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    fasta_records = list(SeqIO.parse(FASTA, "fasta"))
    id_by_md5 = {}
    seq_by_id = {}
    for rec in fasta_records:
        seq = str(rec.seq).upper().replace("*", "")
        m = md5(seq)
        id_by_md5[m] = rec.id
        seq_by_id[rec.id] = seq

    pdb_files = [p for p in os.listdir(PDB_DIR) if p.lower().endswith(".pdb")]
    hit_ids = set()
    orphans = []

    for fname in pdb_files:
        fpath = os.path.join(PDB_DIR, fname)
        chain_map = seq_from_pdb(fpath)
        matched = False
        for ch, ch_seq in chain_map.items():
            m = md5(ch_seq)
            seq_id = id_by_md5.get(m)
            if seq_id:
                link_name = f"{seq_id}.pdb" if ch == "A" and list(chain_map.keys()) == ["A"] else f"{seq_id}_chain{ch}.pdb"
                link_path = os.path.join(OUT_DIR, link_name)
                if not os.path.exists(link_path):
                    os.symlink(fpath, link_path)
                hit_ids.add(seq_id)
                matched = True
        if not matched:
            orphans.append(fname)

    missing = [sid for sid in seq_by_id.keys() if sid not in hit_ids]

    print("=== mapping finish ===")
    print(f"PDBs: {len(pdb_files)}")
    print(f"sequences mapped to FASTA: {len(hit_ids)}")
    print(f"mismapping: {len(orphans)}")
    if orphans:
        print("  The example does not match the PDB (top 10):", orphans[:10])
    print(f"The sequences from PDB are still missing in FASTA: {len(missing)}")
    if missing:
        print("  Example missing seq_id (first 10):", missing[:10])
    print(f"mapping dir: {OUT_DIR}")
    print("next: cotton.py use --af3-pdb-dir to the mapping dir")

if __name__ == "__main__":
    main()