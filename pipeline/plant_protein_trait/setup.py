"""Dedicated installation with persistent settings; no global Conda changes."""
import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from .core import digest, read_json, run_lock, utc, write_json


def command(argv, capture=False):
    print("Running:", " ".join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), check=True, text=True,
                          capture_output=capture)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path, help="Dedicated installation directory outside this checkout")
    p.add_argument("--conda", help="Conda executable (otherwise CONDA_EXE or PATH)")
    p.add_argument("--with-databases", action="store_true", help="After environment setup, run the database script")
    p.add_argument("--database", choices=["nr", "swissprot"], default="nr")
    p.add_argument("--interpro-version", default="5.78-109.0")
    p.add_argument("--with-pfam", action="store_true")
    p.add_argument("--pfam-version", default="38.0")
    p.add_argument("--execute", action="store_true", help="Without this flag, only print a plan")
    return p


def install(args):
    package = Path(__file__).resolve().parents[1]
    root = args.root.expanduser().resolve()
    if root == package or root.is_relative_to(package) or package.is_relative_to(root):
        raise ValueError("Choose a dedicated installation directory separate from the pipeline checkout")
    env_file = package / "environment.local.yml"
    env = root / "env"
    spec = {"environment_file_sha256": digest(env_file.read_bytes()), "prefix": str(env),
            "platform": "linux-x86_64"}
    db_args = ["--root", str(root / "resources"), "--database", args.database,
               "--interpro-version", args.interpro_version, "--pfam-version", args.pfam_version]
    if args.with_pfam:
        db_args.append("--with-pfam")
    print(json.dumps({"environment": str(env), "environment_definition": str(env_file),
                      "prepare_databases": args.with_databases, "database_arguments": db_args}, indent=2))
    if not args.execute:
        print("PLAN ONLY. Add --execute to install. Conda is a prerequisite; no root access is needed.")
        return root
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ValueError("This local distribution requires Linux x86_64; ARM/aarch64 is not supported")
    conda = args.conda or os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise ValueError("Install Conda/Miniforge first, or pass --conda /path/to/bin/conda")
    with run_lock(root):
        request = root / "installation_request.json"
        if request.exists():
            if read_json(request) != spec:
                raise ValueError("Environment definition changed; choose a new installation root")
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Installation root is not empty; choose a fresh dedicated directory")
        else:
            write_json(request, spec)
        marker = root / "environment_ready.json"
        write_json(root / "installation_status.json", {"status": "running", "updated_utc": utc()})
        try:
            if not marker.exists():
                # Updating an interrupted prefix is explicit, limited to our recorded dedicated root.
                action = "update" if (env / "conda-meta/history").exists() else "create"
                command([conda, "env", action, "--prefix", env, "--file", env_file] +
                        (["--yes"] if action == "create" else []))
            check = command([conda, "run", "--no-capture-output", "--prefix", env, "python", "-c",
                    "import sys,Bio,shutil; assert sys.version_info[:2]==(3,12); "
                    "assert Bio.__version__=='1.85'; "
                    "assert all(shutil.which(x) for x in ['blastp','blastdbcmd','update_blastdb.pl','hmmscan','hmmpress','java','curl'])"], capture=True)
            explicit = command([conda, "list", "--prefix", env, "--explicit"], capture=True)
            (root / "environment.explicit.txt").write_text(explicit.stdout, encoding="utf-8")
            write_json(marker, {**spec, "verified_utc": utc()})
            config = {"schema_version": 1, "pipeline_root": str(package), "conda": str(conda),
                      "environment_prefix": str(env), "python": str(env / "bin/python"),
                      "resources_manifest": str(root / "resources/resources.json")}
            write_json(root / "local_config.json", config)
            if args.with_databases:
                command([conda, "run", "--no-capture-output", "--prefix", env, "python", "-u",
                         package / "0_download_databases.py", *db_args, "--execute"])
            write_json(root / "installation_status.json", {"status": "ready" if args.with_databases else "environment_ready",
                       "updated_utc": utc(), "config": str(root / "local_config.json")})
        except Exception as exc:
            write_json(root / "installation_status.json", {"status": "failed", "error": str(exc), "updated_utc": utc()})
            raise
    print("INSTALLATION_READY" if args.with_databases else "ENVIRONMENT_READY", root / "local_config.json")
    return root


def main(argv=None):
    try:
        install(parser().parse_args(argv))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from None
