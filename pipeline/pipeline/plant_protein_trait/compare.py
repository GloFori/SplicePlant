"""Compare actual normalized evidence; never assert equivalence from matching filenames."""
import argparse
from pathlib import Path

from .core import canonical, digest, fasta, read_json, verify_manifest, write_json
from .inference import evidence_packet
from .parsers import load_annotation


def compare(left, right):
    packets, settings, versions = [], [], []
    for folder in map(Path, [left, right]):
        verify_manifest(folder, read_json(folder / "evidence_manifest.json"))
        query = fasta(folder / "query.fasta")
        packet = evidence_packet(query, load_annotation(folder, query))
        packets.append(packet)
        settings.append(read_json(folder / "request.json")["analysis_settings"])
        versions.append(read_json(folder / "observed_versions.json"))
    if packets[0]["query"]["sequence_sha256"] != packets[1]["query"]["sequence_sha256"]:
        raise ValueError("Different query proteins")
    complete = all(p["stage_status"].get("blast") in {"completed_hits", "completed_no_hits"}
                   and p["stage_status"].get("domains") == "completed"
                   and p["stage_status"].get("go") == "completed" for p in packets)
    hashes = [digest(canonical(p)) for p in packets]
    same = hashes[0] == hashes[1]
    return {"status": "incomplete" if not complete else "matched" if same and settings[0] == settings[1] else "different",
            "evidence_identical": same, "analysis_settings_identical": settings[0] == settings[1],
            "observed_versions_identical": versions[0] == versions[1],
            "left_evidence_sha256": hashes[0], "right_evidence_sha256": hashes[1],
            "observed_versions": versions,
            "limits": "Evidence equality is measured, not guaranteed for live services. Unknown releases remain unknown. "
                      "Hosted model responses can vary even with identical prompts; archive original responses."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    result = compare(args.left, args.right)
    write_json(args.output, result)
    print(result["status"])
    if result["status"] != "matched":
        raise SystemExit(1)
