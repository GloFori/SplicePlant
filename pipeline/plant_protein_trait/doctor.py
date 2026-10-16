#!/usr/bin/env python3
"""Read-only preflight. Never outputs credential values."""
import json
import os
import platform
import shutil
import sys


def main(argv=None):
    report = {"python": platform.python_version(), "supported": sys.version_info >= (3, 10),
              "core_python_dependencies": "standard library only",
              "optional_tools": {x: shutil.which(x) for x in ["blastp", "hmmscan", "interproscan.sh", "TMalign", "foldseek", "ollama"]},
              "credential_configured": {x: bool(os.getenv(x)) for x in ["DASHSCOPE_API_KEY", "OPENAI_API_KEY"]},
              "note": "This is a PATH check. InterProScan may be outside PATH and resolved through --config/--resources. Executable presence does not confirm databases, a running service, models, or GPU capacity."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
