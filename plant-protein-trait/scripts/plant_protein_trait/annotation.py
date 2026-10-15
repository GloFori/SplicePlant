"""Objective protein annotation with explicit online opt-in and frozen evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path

from .core import fasta, fresh_dir, manifest, read_json, write_json, write_tsv
from .net import request, snapshot
from .parsers import parse_blast, parse_domtbl, parse_interpro

BLAST = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
IPR = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"


def external(command: list[str], raw: Path, label: str) -> None:
    """No shell interpolation. Tool failure is a failure, never an empty successful search."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    (raw / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (raw / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    write_json(raw / f"{label}.command.json", {"command": command, "exit_code": result.returncode})
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}); see raw stderr")


def remote_blast(query: dict, raw: Path, database: str, email: str, wait_seconds: int) -> Path:
    params = {"CMD": "Put", "PROGRAM": "blastp", "DATABASE": database,
              "QUERY": ">query_001\n" + query["sequence"], "EXPECT": "1e-5",
              "HITLIST_SIZE": "20", "FILTER": "L", "tool": "plant-protein-trait", "email": email}
    body = request(BLAST, form=params)
    snapshot(raw, "blast_submission.txt", body, BLAST,
             database=database, query_sha256=query["sequence_sha256"])
    match = re.search(r"RID\s*=\s*(\S+)", body.decode())
    if not match:
        raise RuntimeError("NCBI did not return a request ID; inspect raw/blast_submission.txt")
    rid = match.group(1)
    write_json(raw / "blast_job.json", {"rid": rid, "database": database, "status": "submitted"})
    deadline = time.monotonic() + wait_seconds
    poll = 0
    while time.monotonic() + 60 <= deadline:
        time.sleep(60)  # NCBI: poll a RID no more than once per minute.
        poll += 1
        url = BLAST + "?" + urllib.parse.urlencode({"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML2_S"})
        body = request(url)
        snapshot(raw, f"blast_poll_{poll:03d}.txt", body, url)
        if b"<BlastXML2" in body or b"<BlastOutput" in body or b"<BlastOutput2" in body:
            return snapshot(raw, "blast.xml", body, url, database=database, rid=rid)
        if b"Status=FAILED" in body or b"Status=UNKNOWN" in body:
            raise RuntimeError(f"NCBI search failed or expired; RID {rid}")
    raise RuntimeError(f"BLAST still pending after {wait_seconds}s; RID {rid} saved in raw/blast_job.json")


def remote_interpro(query: dict, raw: Path, email: str, wait_seconds: int) -> Path:
    body = request(IPR + "/run", form={"email": email, "title": "single_protein",
                   "sequence": ">query_001\n" + query["sequence"], "goterms": "true", "pathways": "true"})
    job = body.decode().strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
        raise RuntimeError("Invalid InterProScan job ID")
    snapshot(raw, "interpro_submission.txt", body, IPR + "/run", query_sha256=query["sequence_sha256"])
    write_json(raw / "interpro_job.json", {"job_id": job, "status": "submitted"})
    deadline = time.monotonic() + wait_seconds
    poll = 0
    while time.monotonic() + 15 <= deadline:
        time.sleep(15)
        poll += 1
        status = request(IPR + "/status/" + job).decode().strip()
        snapshot(raw, f"interpro_status_{poll:03d}.txt", status.encode(), IPR + "/status/" + job)
        if status == "FINISHED":
            url = IPR + "/result/" + job + "/json"
            return snapshot(raw, "interpro.json", request(url), url, job_id=job)
        if status in {"ERROR", "FAILURE", "NOT_FOUND"}:
            raise RuntimeError(f"InterProScan status {status}; job {job}")
    raise RuntimeError(f"InterProScan pending; job {job} saved in raw/interpro_job.json")


def pdb_plddt(path: Path) -> dict:
    residues = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA" and line[16:17] in {" ", "A"}:
            residues[(line[21:22], line[22:27])] = float(line[60:66])
    vals = list(residues.values())
    if not vals or any(not 0 <= x <= 100 for x in vals):
        raise ValueError("Predicted PDB lacks valid per-residue pLDDT in CA B-factor fields")
    return {"avg_plddt": sum(vals) / len(vals), "residues_scored": len(vals),
            "fraction_plddt_below_70": sum(x < 70 for x in vals) / len(vals),
            "plddt_method": "one CA per residue, first model"}


def afdb(query: dict, accession: str, raw: Path) -> dict:
    if not re.fullmatch(r"[A-Z0-9]+(?:-\d+)?", accession):
        raise ValueError("Invalid UniProt accession")
    url = "https://alphafold.ebi.ac.uk/api/prediction/" + accession
    data = request(url)
    snapshot(raw, "afdb.json", data, url)
    models = json.loads(data)
    if isinstance(models, dict):
        models = [models]
    exact = [m for m in models if m.get("uniprotSequence", "").upper() == query["sequence"]
             and m.get("uniprotStart", 1) == 1 and m.get("uniprotEnd", query["length"]) == query["length"]]
    if not exact:
        return {"status": "reference_only", "source": "AlphaFold DB", "accession": accession,
                "avg_plddt": None, "tm_score": None,
                "reason": "No exact full-length sequence match; no target structure evidence assigned"}
    model = exact[0]
    download = model.get("pdbUrl")
    if not download:
        return {"status": "missing", "avg_plddt": None, "tm_score": None,
                "reason": "AFDB returned no PDB download URL"}
    parsed = urllib.parse.urlsplit(download)
    if parsed.hostname != "alphafold.ebi.ac.uk" or parsed.scheme != "https":
        raise ValueError("Unexpected AFDB download host")
    path = snapshot(raw, "target_model.pdb", request(download), download)
    quality = pdb_plddt(path)
    if quality["residues_scored"] != query["length"]:
        raise ValueError("AFDB PDB residue count does not match the verified full-length query")
    return {"status": "available", "source": "AlphaFold DB prediction; exact full-length sequence match",
            "accession": accession, "pdb_path": "raw/target_model.pdb", "model_id": model.get("entryId"),
            **quality, "tm_score": None, "foldseek": "not_run",
            "caution": "pLDDT is local structural confidence, not evidence of a trait"}


def collect(args) -> Path:
    query = fasta(args.fasta)
    if (args.online_blast or args.interpro_remote) and not args.email:
        raise ValueError("--email is required for online BLAST / InterProScan; no default contact identity")
    if args.online_blast and (args.blast_xml or args.blast_db):
        raise ValueError("Choose one BLAST source: local database, existing XML, or online search")
    if args.blast_xml and args.blast_db:
        raise ValueError("Choose --blast-xml or --blast-db")
    if args.interpro_json and args.interpro_remote:
        raise ValueError("Choose --interpro-json or --interpro-remote")
    out = fresh_dir(args.output)
    raw = out / "raw"
    raw.mkdir()
    write_json(out / "sequence_qc.json", {k: v for k, v in query.items() if k != "sequence"})
    (out / "query.fasta").write_text(">" + query["id"] + "\n" + query["sequence"] + "\n", encoding="utf-8")
    rec = {"seq_id": query["id"], "length": query["length"],
           "sequence_sha256": query["sequence_sha256"], "blast_hits": [], "domains": [], "go_terms": [],
           "structure": {"status": "missing", "avg_plddt": None, "tm_score": None,
                         "foldseek": "not_run", "reason": "No target structure provided"},
           "status": {"blast": "not_run", "domains": "not_run", "go": "not_run"}}
    def imported(source: str, name: str) -> Path:
        return snapshot(raw, name, Path(source).read_bytes(), "user_supplied_file",
                        original_basename=Path(source).name)
    try:
        blast_xml = None
        if args.blast_xml:
            blast_xml = imported(args.blast_xml, "blast.xml")
        elif args.blast_db:
            blast_xml = raw / "blast.xml"
            external([args.blastp, "-query", str(out / "query.fasta"), "-db", args.blast_db,
                      "-out", str(blast_xml), "-outfmt", "5", "-evalue", "1e-5", "-seg", "yes",
                      "-max_target_seqs", "20", "-num_threads", str(args.threads)], raw, "blastp")
        elif args.online_blast:
            blast_xml = remote_blast(query, raw, args.remote_database, args.email, args.max_wait)
        if blast_xml:
            rec["blast_hits"], rec["blast_metadata"] = parse_blast(blast_xml, query)
            rec["status"]["blast"] = "completed_hits" if rec["blast_hits"] else "completed_no_hits"
        interpro_json = None
        if args.interpro_json:
            interpro_json = imported(args.interpro_json, "interpro.json")
        elif args.interpro_remote:
            interpro_json = remote_interpro(query, raw, args.email, args.max_wait)
        if interpro_json:
            rec["domains"], rec["go_terms"] = parse_interpro(interpro_json, query)
            rec["status"]["domains"] = "completed"
            rec["status"]["go"] = "completed"
        if args.pfam_db:
            domtbl = raw / "pfam.domtbl"
            external([args.hmmscan, "--cpu", str(args.threads), "--domtblout", str(domtbl),
                      args.pfam_db, str(out / "query.fasta")], raw, "hmmscan")
            rec["domains"].extend(parse_domtbl(domtbl, query))
            rec["status"]["domains"] = "completed"
        if args.afdb_accession:
            rec["structure"] = afdb(query, args.afdb_accession, raw)
    except Exception as exc:
        write_json(out / "run_status.json", {"status": "failed", "error": str(exc),
                   "note": "Partial raw artifacts retained; no completed evidence manifest generated"})
        raise
    write_json(out / "annotation_full.json", {query["id"]: rec})
    write_tsv(out / "annotation_summary.tsv", [{"seq_id": query["id"], "length": query["length"],
              "num_blast_hits": len(rec["blast_hits"]), "num_domains": len(rec["domains"]),
              "num_go_terms": len(rec["go_terms"]), "structure_status": rec["structure"]["status"]}],
              ["seq_id", "length", "num_blast_hits", "num_domains", "num_go_terms", "structure_status"])
    write_tsv(out / "blast_hits.tsv", rec["blast_hits"], ["accession", "description", "organism", "identity_pct",
              "query_coverage_pct", "query_coverage_union_pct", "e_value", "bit_score", "statistics_scope"])
    write_tsv(out / "domains.tsv", rec["domains"], ["accession", "name", "database", "interpro_accession",
              "interpro_name", "e_value", "score", "annotation_method"])
    write_tsv(out / "go.tsv", rec["go_terms"], ["id", "name", "category", "annotation_method", "experimental_on_query"])
    write_json(out / "structure_info.json", {query["id"]: rec["structure"]})
    write_tsv(out / "structure_summary.tsv", [{"seq_id": query["id"], **rec["structure"]}],
              ["seq_id", "status", "source", "avg_plddt", "tm_score", "reason"])
    (out / "pipeline_summary.txt").write_text(
        "Single-protein evidence collection. No weighted functional/trait score is computed.\n" +
        json.dumps(rec["status"], indent=2) + "\n", encoding="utf-8")
    write_json(out / "run_status.json", {"status": "completed", "stages": rec["status"]})
    names = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    write_json(out / "evidence_manifest.json", manifest(out, names,
               sequence_sha256=query["sequence_sha256"], schema_version="1.0", pipeline_version="0.1.0"))
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect independent single-protein evidence; default is offline")
    p.add_argument("-i", "--input", "--fasta", dest="fasta", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--blast-xml", help="Existing XML or XML2; aligned query residues are checked")
    p.add_argument("--blast-db", help="Existing local BLAST database prefix; never downloaded automatically")
    p.add_argument("--blastp", default="blastp")
    p.add_argument("--online-blast", action="store_true")
    p.add_argument("--remote-database", default="swissprot")
    p.add_argument("--interpro-json")
    p.add_argument("--interpro-remote", action="store_true")
    p.add_argument("--pfam-db", help="Existing pressed Pfam HMM database")
    p.add_argument("--hmmscan", default="hmmscan")
    p.add_argument("--afdb-accession", help="Explicit UniProt accession; must match the whole target sequence")
    p.add_argument("--email", help="Your contact for NCBI/EMBL-EBI; no address is bundled")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-wait", type=int, default=1800)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or args.max_wait < 60:
        raise SystemExit("--threads must be >=1; --max-wait must be >=60")
    try:
        print(collect(args))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from None
