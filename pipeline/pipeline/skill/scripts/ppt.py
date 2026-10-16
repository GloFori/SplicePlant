#!/usr/bin/env python3
"""Bind an installed Skill to the one maintained pipeline; never duplicate its engine."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]


def locate(explicit=None):
    value = explicit or os.environ.get("PPT_PIPELINE_ROOT")
    if not value and (SKILL / "pipeline_binding.json").is_file():
        value = json.loads((SKILL / "pipeline_binding.json").read_text(encoding="utf-8"))["pipeline_root"]
    root = Path(value).expanduser().resolve() if value else SKILL.parent
    required = ["llm.py", "0_install_environment.py", "0_download_databases.py", "4_cotton.py",
                "plant_protein_trait/__init__.py", "README.md"]
    if not all((root / name).is_file() for name in required):
        raise ValueError("Cannot locate pipeline. Pass --pipeline-root /path/to/SplicePlant/pipeline or set PPT_PIPELINE_ROOT")
    return root


def install_skill(root, destination):
    destination = Path(destination).expanduser().resolve()
    if destination == SKILL or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Install the Skill outside the source checkout")
    if destination.exists():
        raise ValueError("Skill destination already exists; choose a new directory or review the existing installation")
    shutil.copytree(root / "skill", destination,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "pipeline_binding.json"))
    (destination / "pipeline_binding.json").write_text(json.dumps({"pipeline_root": str(root)}, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-root")
    p.add_argument("command", choices=["locate", "install", "setup", "download", "run", "stage"])
    args, rest = p.parse_known_args(argv)
    root = locate(args.pipeline_root)
    if args.command == "locate":
        print(root)
        return
    if args.command == "install":
        sub = argparse.ArgumentParser(description="Copy Skill and record the pipeline location")
        sub.add_argument("--destination", required=True)
        target = sub.parse_args(rest).destination
        print(install_skill(root, target))
        return
    scripts = {"setup": "0_install_environment.py", "download": "0_download_databases.py",
               "run": "llm.py", "stage": "llm.py"}
    extra = ["pipeline"] if args.command == "run" else []
    command = [sys.executable, str(root / scripts[args.command]), *extra, *rest]
    # An installed Skill may run in a different Python environment from the pipeline.
    # Use the recorded Conda environment when supplied, including its activation variables.
    if args.command == "run":
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--config")
        options, _ = probe.parse_known_args(rest)
        if options.config:
            config = json.loads(Path(options.config).read_text(encoding="utf-8"))
            command = [config["conda"], "run", "--no-capture-output", "--prefix", config["environment_prefix"],
                       "python", str(root / "llm.py"), "pipeline", *rest]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, IndexError) as exc:
        raise SystemExit(str(exc)) from None
