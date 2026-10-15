#!/usr/bin/env python3
import argparse
import json
from plant_protein_trait.core import fasta, write_json


def main():
    p = argparse.ArgumentParser(description="Validate exactly one unmodified canonical amino-acid sequence")
    p.add_argument("--fasta", required=True)
    p.add_argument("--expected-id")
    p.add_argument("--expected-length", type=int)
    p.add_argument("--output")
    args = p.parse_args()
    rec = fasta(args.fasta)
    if args.expected_id and rec["id"] != args.expected_id:
        raise SystemExit("Unexpected sequence ID")
    if args.expected_length is not None and rec["length"] != args.expected_length:
        raise SystemExit("Unexpected sequence length")
    result = {k: v for k, v in rec.items() if k != "sequence"}
    result["status"] = "QC_PASS"
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
