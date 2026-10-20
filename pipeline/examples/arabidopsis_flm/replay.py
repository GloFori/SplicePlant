#!/usr/bin/env python3
"""Replay the supplied FLM annotations with the adjacent v0.3.0 pipeline."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

EXAMPLE = Path(__file__).resolve().parent
CASES = ("FLM_wt", "FLM_alt")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_package():
    manifest = load(EXAMPLE.parent / "manifest.json")
    root = EXAMPLE.parent.resolve()
    for name, expected in manifest["files"].items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Missing or invalid archive path: {name}")
        if sha(candidate) != expected:
            raise ValueError(f"Archive checksum mismatch: {name}")
    print(f"PASS: {len(manifest['files'])} published files verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check published file hashes without the pipeline")
    parser.add_argument("--out", type=Path, help="New or empty run directory outside examples/")
    parser.add_argument("--pipeline-dir", type=Path, default=EXAMPLE.parents[1], help="Directory containing 4_cotton.py and plant_protein_trait/")
    args = parser.parse_args()
    verify_package()
    if args.verify_only:
        if args.out:
            parser.error("Use --verify-only or --out, not both")
        return
    if not args.out:
        parser.error("--out is required unless --verify-only is used")

    pipeline = args.pipeline_dir.resolve()
    if not (pipeline / "4_cotton.py").is_file() or not (pipeline / "plant_protein_trait").is_dir():
        raise ValueError("Place examples/ inside pipeline/, or provide --pipeline-dir /path/to/pipeline")
    sys.path.insert(0, str(pipeline))
    from plant_protein_trait import __version__
    from plant_protein_trait.core import fasta, verify_manifest

    if __version__ != "0.3.0":
        raise ValueError(f"This reference replay targets pipeline 0.3.0; found {__version__}")

    out = args.out.resolve()
    if out.is_relative_to(EXAMPLE.parent.resolve()):
        raise ValueError("Use an output directory outside examples/ to preserve the published archive")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be a new or empty directory; existing results are never overwritten")

    raw_xml = EXAMPLE / "archived_annotation" / "blast_results.xml"
    raw_ipr = EXAMPLE / "archived_annotation" / "interproscan_results.json"
    source = ET.parse(raw_xml).getroot()
    if source.tag != "BlastOutput":
        raise ValueError("Expected the archived XML1 BLAST output")
    queries = {name: fasta(EXAMPLE / "input" / (name + "_pep.fa")) for name in CASES}
    selected = {}
    for name, query in queries.items():
        matches = [it for it in source.findall(".//Iteration")
                   if it.findtext("Iteration_query-def", "").split()[:1] == [query["id"]]]
        if len(matches) != 1 or int(matches[0].findtext("Iteration_query-len", "0")) != query["length"]:
            raise ValueError(f"Missing, ambiguous or wrong-length archived BLAST query for {name}")
        selected[name] = matches[0]

    out.mkdir(parents=True, exist_ok=True)
    split_dir = out / "split_blast"
    split_dir.mkdir()
    state = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": __version__, "python_version": sys.version.split()[0],
        "scope": "archived_annotation_replay", "database_searches_performed": False,
        "model_calls_performed_in_replay": False,
        "paper_model_analysis": "completed previously, as confirmed by the author; supplied report preserved",
        "paper_report_sha256": sha(EXAMPLE / "paper_model_output" / "case2.md"),
        "cases": {},
        "code_sha256": {p.relative_to(pipeline).as_posix(): sha(p)
                        for p in sorted((pipeline / "plant_protein_trait").glob("*.py"))},
    }
    state["code_sha256"]["4_cotton.py"] = sha(pipeline / "4_cotton.py")
    save(out / "replay_status.json", state)
    derivations = []
    try:
        for name, query in queries.items():
            derived = deepcopy(source)
            container = derived.find("BlastOutput_iterations")
            if container is None:
                raise ValueError("BLAST XML lacks its iteration container")
            container.clear()
            iteration = deepcopy(selected[name])
            container.append(iteration)
            for suffix in ("ID", "def", "len"):
                header = derived.find("BlastOutput_query-" + suffix)
                if header is not None:
                    header.text = iteration.findtext("Iteration_query-" + suffix)
            xml = split_dir / (name + ".xml")
            ET.ElementTree(derived).write(xml, encoding="utf-8", xml_declaration=True)
            derivations.append({
                "query": name, "source_sha256": sha(raw_xml), "derived_sha256": sha(xml),
                "method": "Select the matching Iteration and update root query header fields; retain hits, HSPs and database statistics",
            })
            annotation = out / name / "annotation"
            command = [sys.executable, str(pipeline / "4_cotton.py"),
                       "--fasta", str(EXAMPLE / "input" / (name + "_pep.fa")),
                       "--blast-xml", str(xml), "--interpro-json", str(raw_ipr),
                       "--remote-database", "refseq_protein", "--output", str(annotation)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120)
            (out / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"Annotation import failed for {name}; see {name}.log")
            verify_manifest(annotation, load(annotation / "evidence_manifest.json"))
            status = load(annotation / "run_status.json")
            if status.get("status") != "completed":
                raise RuntimeError(f"Annotation import did not complete for {name}")
            actual = load(annotation / "annotation_full.json")
            expected = load(EXAMPLE / "expected_replay" / name / "annotation_full.json")
            if actual != expected:
                raise ValueError(f"Parsed annotation differs from the bundled reference for {name}; inspect output and code versions")
            record = actual[name]
            state["cases"][name] = {
                "status": "matched", "protein_length": query["length"],
                "sequence_sha256": query["sequence_sha256"],
                "blast_hits": len(record["blast_hits"]), "domain_records": len(record["domains"]),
                "go_terms": len(record["go_terms"]), "evidence_manifest_verified": True,
            }
            print(f"PASS: {name}: {query['length']} aa; {len(record['blast_hits'])} BLAST hits, "
                  f"{len(record['domains'])} domain records, {len(record['go_terms'])} GO terms; full record matched.")
            save(out / "replay_status.json", state)
        save(out / "split_blast_provenance.json", derivations)
        state["status"] = "completed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(out / "replay_status.json", state)
        print("PASS: both annotations match the bundled reference records; no database searches or model calls performed.")
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        save(out / "replay_status.json", state)
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, ET.ParseError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
