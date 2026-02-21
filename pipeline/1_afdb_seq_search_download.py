import csv
import argparse
import os
import re
import sys
import time
import json
import textwrap
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import requests
def safe_tsv_head_to_markdown(path: Path, max_rows: int = 20, max_cols: int = 16) -> str:
    if not path.exists():
        return "_(missing)_"
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            safe_row = [str(c) if c is not None else "" for c in row]
            rows.append(safe_row)
            if i >= max_rows:
                break
    if not rows:
        return "_(empty)_"
    header = rows[0]
    data = rows[1:]
    header = header[:max_cols]
    data = [r[:max_cols] + [""] * max(0, len(header) - len(r[:max_cols])) for r in data]
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")
    md = []
    md.append("| " + " | ".join(esc(h) for h in header) + " |")
    md.append("| " + " | ".join("---" for _ in header) + " |")
    for r in data:
        md.append("| " + " | ".join(esc(x) for x in r) + " |")
    return "\n".join(md)
def tsv_head_or_fallback(path: Path, label: str, max_rows: int = 20, max_cols: int = 16) -> str:
    try:
        return safe_tsv_head_to_markdown(path, max_rows=max_rows, max_cols=max_cols)
    except Exception as e:
        return f"_(failed to preview {label}: {e})_"
def is_uniprot_acc(s: str) -> bool:
    if not s or len(s) not in (6, 10):
        return False
    if not s[0].isalpha() or s.upper() != s or not s.isalnum():
        return False
    if not any(ch.isdigit() for ch in s):
        return False
    return True
AF_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
NCBI_BLAST = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
UNI_ACC_RE = re.compile(r"\|([A-NR-Z0-9]{6,10})\|")
FASTA_HDR = re.compile(r"^>\S+")

def read_sequence(fa_or_seq: str) -> str:
    s = None
    try:
        p = Path(fa_or_seq)
        if p.exists():
            seq = []
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if FASTA_HDR.match(line):
                        if seq:
                            break
                        continue
                    seq.append(line)
            s = "".join(seq).upper()
    except OSError:
        s = None

    if s is None:
        s = "".join(
            x.strip() for x in fa_or_seq.splitlines()
            if not FASTA_HDR.match(x)
        ).upper()

    if (not s) or any(ch for ch in s if ch not in "ACDEFGHIKLMNPQRSTVWYBXZJUO*"):
        raise ValueError("Not a legitimate amino acid sequence")

    return s
def ncbi_blast_sequence(seq: str, program="blastp", database="swissprot",
                        expect=1e-5, hitlist_size=100, sleep_s=5, max_wait_s=300) -> str:
    put_data = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": database,
        "QUERY": seq,
        "HITLIST_SIZE": str(hitlist_size),
        "EXPECT": str(expect),
    }
    r = requests.post(NCBI_BLAST, data=put_data, timeout=60)
    r.raise_for_status()
    m_rid = re.search(r"RID = (\S+)", r.text)
    m_rtoe = re.search(r"RTOE = (\d+)", r.text)
    if not m_rid:
        raise RuntimeError(f"BLAST submission failed. Unable to parse RID. \n---\n{r.text}\n---")
    rid = m_rid.group(1)
    rtoe = int(m_rtoe.group(1)) if m_rtoe else 10

    waited = 0
    while waited <= max_wait_s:
        time.sleep(max(sleep_s, rtoe))
        get_params = {"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"}
        g = requests.get(NCBI_BLAST, params=get_params, timeout=60)
        g.raise_for_status()
        txt = g.text
        if "Status=FAILED" in txt:
            raise RuntimeError("BLAST task failed")
        if "Status=UNKNOWN" in txt:
            raise RuntimeError("BLAST task is unknown (RID has expired or is invalid)")
        if "Status=READY" in txt and "ThereAreHits=yes" in txt:
            return txt
        if "Status=READY" in txt and "ThereAreHits=no" in txt:
            return ""
        waited += sleep_s
    raise TimeoutError("BLAST result waiting timeout")

def local_blast_sequence(seq: str, db_prefix: str, blastp_path: str = "blastp",
                         evalue: float = 1e-5, max_target_seqs: int = 100,
                         threads: int = 1, timeout_s: int = 1800) -> str:
    with tempfile.TemporaryDirectory() as td:
        qfa = Path(td) / "query.fa"
        qfa.write_text(">q\n" + seq + "\n", encoding="utf-8")

        cmd = [
            blastp_path,
            "-query", str(qfa),
            "-db", db_prefix,
            "-outfmt", "5",
            "-evalue", str(evalue),
            "-max_target_seqs", str(max_target_seqs),
            "-num_threads", str(threads),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, timeout=timeout_s, check=False)
        except FileNotFoundError:
            raise RuntimeError(f"Cannot find the blastp executable file：{blastp_path}")
        if res.returncode != 0:
            raise RuntimeError(f"local blastp failed（{res.returncode}）\nCMD: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")
        xml = res.stdout or ""
        return xml

def extract_uniprot_accessions_from_blast_xml(xml_text: str, top_k: int = 50) -> List[str]:
    if not xml_text:
        return []
    cand = set()
    strict_pipe = re.compile(r"\b(?:sp|tr)\|([A-Z][A-Z0-9]{5}(?:[A-Z0-9]{4})?)\|")
    for m in re.finditer(r"<Hit_id>([^<]+)</Hit_id>", xml_text):
        for mm in strict_pipe.finditer(m.group(1)):
            acc = mm.group(1)
            if is_uniprot_acc(acc):
                cand.add(acc)
    for m in re.finditer(r"<Hit_def>([^<]+)</Hit_def>", xml_text):
        for mm in strict_pipe.finditer(m.group(1)):
            acc = mm.group(1)
            if is_uniprot_acc(acc):
                cand.add(acc)
    out = sorted(cand)
    return out[:top_k]

def afdb_get_models_for_accession(acc: str) -> List[Dict]:
    url = f"{AF_API_BASE}/{acc}"
    r = requests.get(url, timeout=60)
    if r.status_code in (400, 404):
        return []
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]
    return data or []

def pick_best_download_url(entry: Dict, prefer: str = "pdb") -> Optional[Tuple[str, str]]:
    keys = {
        "pdb": ["pdbUrl", "pdb_url"],
        "cif": ["cifUrl", "cif_url", "mmCifUrl", "mmcifUrl"],
        "bcif": ["bcifUrl", "bcif_url"]
    }
    order = [prefer, "pdb", "cif", "bcif"] if prefer != "pdb" else ["pdb", "cif", "bcif"]
    for fmt in order:
        for k in keys.get(fmt, []):
            if k in entry and entry[k]:
                return fmt, entry[k]
    acc = entry.get("uniprotAccession") or entry.get("accession") or entry.get("uniprot_id")
    if acc:
        base = "https://alphafold.ebi.ac.uk/files"
        for v in ("v4", "v3"):
            if prefer == "pdb":
                return "pdb", f"{base}/AF-{acc}-F1-model_{v}.pdb"
            if prefer == "cif":
                return "cif", f"{base}/AF-{acc}-F1-model_{v}.cif"
    return None

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def download_file(url: str, out: Path, overwrite: bool = False) -> bool:
    try:
        if out.exists() and not overwrite:
            return True
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as fh:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
        return True
    except requests.HTTPError:
        return False
def main():
    ap = argparse.ArgumentParser(
        description="Use sequence search and automatically download the AlphaFold DB models (PDB/mmCIF)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--seq_or_fasta", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--prefer", choices=["pdb", "cif", "bcif"], default="pdb")
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--local_db", default=None)
    ap.add_argument("--blastp", default="blastp")
    ap.add_argument("--evalue", type=float, default=1e-5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--online_timeout", type=int, default=300)
    args = ap.parse_args()
    seq = read_sequence(args.seq_or_fasta)
    print(f"[INFO] Sequence length = {len(seq)} aa")
    xml = ""
    if args.local_db:
        print(f"[INFO] Running local BLASTP against DB: {args.local_db}")
        xml = local_blast_sequence(
            seq=seq, db_prefix=args.local_db, blastp_path=args.blastp,
            evalue=args.evalue, max_target_seqs=max(100, args.top_k),
            threads=args.threads, timeout_s=1800
        )
    else:
        print("[INFO] Submitting BLAST (blastp vs swissprot @ NCBI URL API) ...")
        xml = ncbi_blast_sequence(seq, program="blastp", database="swissprot",
                                  expect=args.evalue, hitlist_size=max(100, args.top_k),
                                  sleep_s=5, max_wait_s=int(args.online_timeout))

    accs = extract_uniprot_accessions_from_blast_xml(xml, top_k=args.top_k)
    print(f"[INFO] UniProt candidates = {len(accs)}")
    if not accs:
        print("[WARNING] No UniProt accession was found, so the download from AFDB cannot proceed.")
        sys.exit(2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for acc in accs:
        models = afdb_get_models_for_accession(acc)
        tried_urls = []
        ok = False
        for entry in models or [{}]:
            pick = pick_best_download_url(entry, prefer=args.prefer)
            if not pick:
                continue
            fmt, url = pick
            tried_urls.append(url)
            ext = "pdb" if fmt == "pdb" else ("cif" if fmt == "cif" else "bcif")
            dst = outdir / safe_filename(f"AFDB_{acc}.{ext}")
            if download_file(url, dst, overwrite=False):
                print(f"[OK] {acc} -> {dst}")
                downloaded += 1
                ok = True
                break
        if not ok:
            for v in ("v4", "v3"):
                ext = "pdb" if args.prefer == "pdb" else "cif"
                url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_{v}.{ext}"
                tried_urls.append(url)
                dst = outdir / safe_filename(f"AFDB_{acc}.{ext}")
                if download_file(url, dst, overwrite=False):
                    print(f"[OK] {acc} (fallback {v}) -> {dst}")
                    downloaded += 1
                    ok = True
                    break
        if not ok:
            print(f"[MISS] {acc} cannot download, try：{'; '.join(tried_urls[:3])}{' ...' if len(tried_urls)>3 else ''})")
    if downloaded < args.min_hits:
        print(f"[FAIL] Successful download {downloaded}，less then required {args.min_hits}")
        sys.exit(3)
    print(f"[DONE] Successful download {downloaded} AlphaFold Structure to: {outdir}")
if __name__ == "__main__":
    main()