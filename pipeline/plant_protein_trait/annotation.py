"""Objective protein annotation with explicit online opt-in and frozen evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .core import canonical, digest, fasta, fresh_dir, manifest, read_json, run_lock, utc, verify_manifest, write_json, write_tsv
from .net import request, snapshot
from .parsers import parse_blast, parse_domtbl, parse_interpro
from .jobs import PendingJob, cached, run_job
from .settings import analysis_settings, local_blast_options
from .configuration import configure_resources

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
    with run_lock(args.output):
        return _collect(args)


def _collect(args) -> Path:
    configure_resources(args)
    query = fasta(args.fasta)
    settings = analysis_settings(args)
    if args.blast_rid:
        args.online_blast = True
    if args.interpro_job_id:
        args.interpro_remote = True
    if (args.online_blast or args.interpro_remote) and not args.email:
        raise ValueError("--email is required for online BLAST / InterProScan; no default contact identity")
    if args.online_blast and (args.blast_xml or args.blast_db):
        raise ValueError("Choose one BLAST source: local database, existing XML, or online search")
    if args.blast_xml and args.blast_db:
        raise ValueError("Choose --blast-xml or --blast-db")
    if sum(bool(x) for x in [args.interpro_json, args.interpro_remote, args.interproscan]) > 1:
        raise ValueError("Choose one InterPro source: JSON, remote service, or local InterProScan")
    if args.pfam_db and args.pfam_domtbl:
        raise ValueError("Choose --pfam-db or --pfam-domtbl")
    if args.structure_pdb and (args.afdb_accession or not args.structure_origin):
        raise ValueError("A local PDB requires --structure-origin and must not be combined with --afdb-accession")
    if (args.structure_origin or args.structure_chain) and not args.structure_pdb:
        raise ValueError("Structure origin/chain requires --structure-pdb")
    if args.threads < 1 or args.max_wait < 60 or args.request_timeout < 1:
        raise ValueError("threads/request timeout must be positive; max-wait must be >=60")
    sources = {k: getattr(args, k) for k in ["blast_db", "online_blast", "interpro_remote", "interproscan",
               "pfam_db", "afdb_accession", "blastp", "hmmscan", "database_release", "structure_origin", "structure_chain"]}
    for key in ["blast_xml", "interpro_json", "pfam_domtbl", "structure_pdb"]:
        value = getattr(args, key)
        sources[key] = {"path": str(Path(value).resolve()), "sha256": digest(Path(value).read_bytes())} if value else None
    sources["resources_sha256"] = digest(Path(args.resources).read_bytes()) if args.resources else None
    spec = {"sequence_sha256": query["sequence_sha256"], "sequence_id": query["id"],
            "analysis_settings": settings, "sources": sources}
    if args.resume:
        out = Path(args.output).resolve()
        if not (out / "request.json").is_file():
            raise ValueError("No resumable request.json. For v0.1 jobs use a NEW output directory with "
                             "--blast-rid / --interpro-job-id and the original analysis parameters.")
        if read_json(out / "request.json") != spec:
            raise ValueError("Resume sequence, sources or analysis settings differ from the saved request")
        for kind, key, supplied in [("blast", "rid", args.blast_rid), ("interpro", "job_id", args.interpro_job_id)]:
            job_path = out / "raw" / f"{kind}_job.json"
            if supplied and job_path.exists() and read_json(job_path).get(key) not in {None, supplied}:
                raise ValueError("Supplied job ID differs from the saved run")
        if (out / "evidence_manifest.json").is_file():
            verify_manifest(out, read_json(out / "evidence_manifest.json"))
            return out
    else:
        out = fresh_dir(args.output).resolve()
        write_json(out / "request.json", spec)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
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
    write_json(out / "run_status.json", {"status": "running", "started_utc": utc()})
    try:
        if args.resources:
            imported(args.resources, "resources.json")
        remote_results = {}
        jobs = {"blast": args.online_blast, "interpro": args.interpro_remote}
        errors = []
        if any(jobs.values()):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(run_job, kind, query, raw, settings, args.email, args.max_wait,
                                      args.request_timeout, args.blast_rid if kind == "blast" else args.interpro_job_id): kind
                           for kind, enabled in jobs.items() if enabled}
                for future in as_completed(futures):
                    try:
                        remote_results[futures[future]] = future.result()
                    except Exception as exc:
                        errors.append(exc)
            if errors:
                message = "; ".join(str(e) for e in errors)
                if all(isinstance(e, PendingJob) for e in errors):
                    raise PendingJob(message)
                raise RuntimeError(message)
        blast_xml = None
        if args.blast_xml:
            blast_xml = imported(args.blast_xml, "blast.xml")
        elif args.blast_db:
            blast_xml = raw / "blast.xml"
            if not cached(raw, "blast.xml"):
                print("blast: running locally", flush=True)
                external([args.blastp, "-query", str(out / "query.fasta"), "-db", args.blast_db,
                          "-out", str(blast_xml), "-outfmt", "5", "-num_threads", str(args.threads),
                          *local_blast_options(settings)], raw, "blastp")
                snapshot(raw, "blast.xml", blast_xml.read_bytes(), "local_blastp", settings=settings["blast"])
        elif args.online_blast:
            blast_xml = remote_results["blast"]
        if blast_xml:
            rec["blast_hits"], rec["blast_metadata"] = parse_blast(blast_xml, query)
            rec["status"]["blast"] = "completed_hits" if rec["blast_hits"] else "completed_no_hits"
        interpro_json = None
        if args.interpro_json:
            interpro_json = imported(args.interpro_json, "interpro.json")
        elif args.interpro_remote:
            interpro_json = remote_results["interpro"]
        elif args.interproscan:
            interpro_json = raw / "interpro.json"
            if not cached(raw, "interpro.json"):
                print("interpro: running locally (precalculated web lookup disabled)", flush=True)
                command = [args.interproscan, "-i", str(out / "query.fasta"), "-f", "json", "-o", str(interpro_json),
                           "-goterms", "-pa", "-dp", "-cpu", str(args.threads), "-T", str(raw / "interpro_tmp")]
                if settings["interpro"]["applications"]:
                    command += ["-appl", ",".join(settings["interpro"]["applications"])]
                external(command, raw, "interproscan")
                snapshot(raw, "interpro.json", interpro_json.read_bytes(), "local_interproscan",
                         settings=settings["interpro"])
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
        elif args.pfam_domtbl:
            domtbl = imported(args.pfam_domtbl, "pfam.domtbl")
            rec["domains"].extend(parse_domtbl(domtbl, query))
            rec["status"]["domains"] = "completed"
        if args.afdb_accession:
            rec["structure"] = afdb(query, args.afdb_accession, raw)
        elif args.structure_pdb:
            from .structure_tools import structure_evidence
            pdb = imported(args.structure_pdb, "target_model.pdb")
            rec["structure"] = structure_evidence(pdb, query, args.structure_origin, args.structure_chain)
            rec["structure"]["pdb_path"] = "raw/target_model.pdb"
    except Exception as exc:
        write_json(out / "run_status.json", {"status": "pending" if isinstance(exc, PendingJob) else "failed", "error": str(exc),
                   "note": "Partial raw artifacts retained; no completed evidence manifest generated"})
        raise
    rec["domains"] = sorted(rec["domains"], key=canonical)
    warnings = [read_json(p)["binding_warning"] for p in raw.glob("*_job.json") if read_json(p).get("binding_warning")]
    if warnings:
        rec["binding_warning"] = "; ".join(sorted(set(warnings)))
    versions = {"blast": rec.get("blast_metadata"), "database_release_supplied": args.database_release,
                "interpro_member_databases": sorted({(d.get("database") or "unknown", d.get("database_version") or "unknown")
                                                       for d in rec["domains"]})}
    write_json(out / "observed_versions.json", versions)
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
    names = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and "interpro_tmp" not in p.parts]
    write_json(out / "evidence_manifest.json", manifest(out, names,
               sequence_sha256=query["sequence_sha256"], schema_version="1.0", pipeline_version="0.3.0"))
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect independent single-protein evidence; default is offline")
    p.add_argument("-i", "--input", "--fasta", dest="fasta", required=True)
    p.add_argument("-o", "--output", "--out", required=True)
    p.add_argument("--blast-xml", help="Existing XML or XML2; aligned query residues are checked")
    p.add_argument("--blast-db", help="Existing local BLAST database prefix; never downloaded automatically")
    p.add_argument("--blastp", default="blastp")
    p.add_argument("--online-blast", action="store_true")
    p.add_argument("--remote-database", "--database", help="Database identity; default nr, or the prepared resource manifest")
    p.add_argument("--config", help="local_config.json from 0_install_environment.py")
    p.add_argument("--resources", help="resources.json from 0_download_databases.py; also accepts v0.2 manifests")
    p.add_argument("--use-pfam", action="store_true", help="Include the optional standalone Pfam database recorded in the resources")
    p.add_argument("--database-release", help="Actual local database release/snapshot identifier")
    p.add_argument("--evalue", type=float, default=1e-5)
    p.add_argument("--max-hits", type=int, default=5)
    p.add_argument("--seg", choices=["yes", "no"], default="no")
    p.add_argument("--matrix", default="BLOSUM62")
    p.add_argument("--word-size", type=int, default=3)
    p.add_argument("--gap-open", type=int, default=11)
    p.add_argument("--gap-extend", type=int, default=1)
    p.add_argument("--comp-based-stats", choices=[0, 1, 2, 3], type=int, default=2)
    p.add_argument("--interpro-json")
    p.add_argument("--interpro-remote", action="store_true")
    p.add_argument("--interproscan", help="Local InterProScan 5 executable; uses -dp for local computation")
    p.add_argument("--interpro-applications", default="", help="Comma-separated identical application selection for both routes; empty=all available")
    p.add_argument("--pfam-db", help="Existing pressed Pfam HMM database")
    p.add_argument("--pfam-domtbl", help="Import the same optional standalone Pfam evidence in either route")
    p.add_argument("--hmmscan", default="hmmscan")
    p.add_argument("--afdb-accession", help="Explicit UniProt accession; must match the whole target sequence")
    p.add_argument("--structure-pdb", help="Optional target PDB; full chain sequence is checked")
    p.add_argument("--structure-origin", choices=["alphafold2", "alphafold3", "experimental"])
    p.add_argument("--structure-chain", help="Select a chain when more than one matches the query")
    p.add_argument("--email", help="Your contact for NCBI/EMBL-EBI; no address is bundled")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-wait", type=int, default=1800)
    p.add_argument("--request-timeout", type=int, default=120)
    p.add_argument("--resume", action="store_true", help="Continue this output directory with unchanged sequence/settings")
    p.add_argument("--blast-rid", help="Recover an already submitted NCBI task without submission")
    p.add_argument("--interpro-job-id", help="Recover an already submitted EBI task without submission")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or args.max_wait < 60:
        raise SystemExit("--threads must be >=1; --max-wait must be >=60")
    try:
        print(collect(args))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from None
