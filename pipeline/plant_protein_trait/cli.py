import argparse
import sys
from . import annotation, compare, doctor, inference, report, smoke, workflow
from .core import fasta


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"pipeline": workflow.main, "summarize": report.main, "compare": compare.main,
                "doctor": doctor.main, "smoke-test": smoke.main}
    if not argv or argv[0] in {"-h", "--help"}:
        p = argparse.ArgumentParser(description="Single-protein workflow (upgraded new_llm lineage)")
        p.add_argument("command", choices=["pipeline", "prepare", "run", "freeze", "summarize", "compare", "doctor", "smoke-test", "qc"])
        p.print_help()
        return
    if argv[0] in {"prepare", "run", "freeze"}:
        return inference.main(argv)
    if argv[0] == "qc":
        p = argparse.ArgumentParser(description="Validate one protein FASTA")
        p.add_argument("--fasta", required=True)
        args = p.parse_args(argv[1:])
        q = fasta(args.fasta)
        print(f"QC PASS: {q['id']}, {q['length']} aa, SHA256 {q['sequence_sha256']}")
        return
    if argv[0] not in commands:
        raise SystemExit("Unknown command; run llm.py --help")
    return commands[argv[0]](argv[1:])
