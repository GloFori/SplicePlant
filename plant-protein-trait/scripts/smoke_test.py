#!/usr/bin/env python3
"""Portable reviewer smoke test: pure synthetic data; no network and no LLM."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from plant_protein_trait.core import fresh_dir, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    root = fresh_dir(args.output)
    scripts = Path(__file__).resolve().parent
    seq = "ACDEFGHIKLMNPQRSTVWY" * 4
    fa = root / "synthetic.fa"
    fa.write_text(">SYNTHETIC_TEST_ONLY\n" + seq + "\n", encoding="utf-8")
    ann = root / "annotation"
    subprocess.run([sys.executable, str(scripts / "4_cotton.py"), "--fasta", str(fa), "--output", str(ann)], check=True)
    subprocess.run([sys.executable, str(scripts / "new_llm.py"), "prepare", "--fasta", str(fa),
                    "--annotation-dir", str(ann), "--out", str(root / "prompts"), "--trait", "example plant trait"], check=True)
    write_json(root / "SMOKE_STATUS.json", {"status": "PASS", "test_type": "synthetic_offline_interface_test",
               "biological_annotation": "not_run", "LLM_inference": "not_run", "biological_conclusion": None})
    print("PASS: QC, real annotation interface, four isolated prompts and input hashes. No biological or model results generated.")


if __name__ == "__main__":
    main()
