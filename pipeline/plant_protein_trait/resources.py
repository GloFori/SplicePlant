#!/usr/bin/env python3
"""Prepare local resources once. Without --execute, print the plan and download nothing."""
import argparse
import gzip
import hashlib
import json
import platform
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

from plant_protein_trait.core import read_json, run_lock, utc, write_json


def hash_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args, cwd=None):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_name(path.name + ".part")
        command(["curl", "--fail", "--location", "--retry", "3", "--retry-delay", "10",
                 "--continue-at", "-", "--output", part, url])
        part.replace(path)


def extract_checked(archive, destination):
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Safe archive extraction requires the provided Python 3.12 environment")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path, help="Dedicated resource directory outside the repository")
    p.add_argument("--database", choices=["nr", "swissprot"], default="nr")
    p.add_argument("--interpro-version", default="5.78-109.0")
    p.add_argument("--with-pfam", action="store_true", help="Optional separate HMMER/Pfam supplement")
    p.add_argument("--pfam-version", default="38.0")
    p.add_argument("--execute", action="store_true")
    return p


def inventory(folder):
    return [{"name": f.name, "bytes": f.stat().st_size} for f in sorted(folder.iterdir()) if f.is_file()]


def verify_ready(record):
    if not Path(record["interproscan"]).is_file():
        raise ValueError("Prepared InterProScan executable is missing")
    prefix = Path(record["blast_db"])
    for item in record.get("blast_files", []):
        p = prefix.parent / item["name"]
        if not p.is_file() or p.stat().st_size != item["bytes"]:
            raise ValueError("Prepared BLAST inventory changed; preserve the snapshot and diagnose it")
    subprocess.run(["blastdbcmd", "-db", str(prefix), "-info"], capture_output=True, text=True, check=True)
    if record.get("pfam_db"):
        hmm = Path(record["pfam_db"])
        if not hmm.is_file() or any(not Path(str(hmm) + ext).is_file() for ext in (".h3f", ".h3i", ".h3m", ".h3p")):
            raise ValueError("Prepared standalone Pfam files are missing")


def prepare(args):
    if not re.fullmatch(r"5\.\d+-\d+\.\d+", args.interpro_version) or not re.fullmatch(r"\d+\.\d+", args.pfam_version):
        raise SystemExit("Invalid resource release identifier")
    root = args.root.expanduser().resolve()
    archive_name = f"interproscan-{args.interpro_version}-64-bit.tar.gz"
    ipr_url = f"https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/{args.interpro_version}/{archive_name}"
    pfam_url = f"https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam{args.pfam_version}/Pfam-A.hmm.gz"
    plan = {"database": args.database, "interpro_version": args.interpro_version,
            "interpro_archive": ipr_url, "pfam_archive": pfam_url if args.with_pfam else None,
            "root": str(root), "database_download": ["update_blastdb.pl", "--decompress", args.database]}
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("PLAN ONLY. Check available disk space; nr and full InterProScan are large downloads. Add --execute to proceed.")
        return
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("This pinned InterProScan distribution requires Linux x86_64; use a Linux server/WSL2")
    needed = ["update_blastdb.pl", "blastdbcmd", "blastp", "hmmscan", "curl", "java"] + (["hmmpress"] if args.with_pfam else [])
    missing = [x for x in needed if not shutil.which(x)]
    if missing:
        raise SystemExit("Activate environment.local.yml first; missing: " + ", ".join(missing))
    root.mkdir(parents=True, exist_ok=True)
    plan_path, manifest_path = root / "setup_request.json", root / "resources.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise SystemExit("Resource settings changed; use a new resource directory to preserve the previous snapshot")
    elif any(root.iterdir()):
        raise SystemExit("Resource directory is not empty; use a dedicated directory or configure existing resources directly")
    else:
        write_json(plan_path, plan)
    if manifest_path.exists():
        verify_ready(read_json(manifest_path))
        print("Resources already prepared:", manifest_path)
        return
    db = root / "blast"
    db.mkdir(exist_ok=True)
    checkpoint = root / "blast_ready.json"
    if checkpoint.exists():
        if read_json(checkpoint)["files"] != inventory(db):
            raise ValueError("Partial-install BLAST snapshot changed; use a new directory or restore its files")
    else:
        command(["update_blastdb.pl", "--decompress", args.database], cwd=db)
    info = subprocess.run(["blastdbcmd", "-db", str(db / args.database), "-info"],
                          capture_output=True, text=True, check=True).stdout
    (root / "blast_database_info.txt").write_text(info, encoding="utf-8")
    write_json(checkpoint, {"files": inventory(db), "database_info": info})
    downloads, ipr_dir = root / "downloads", root / "interpro"
    archive = downloads / archive_name
    checksum = downloads / (archive_name + ".md5")
    download(ipr_url, archive)
    download(ipr_url + ".md5", checksum)
    matches = re.findall(r"(?im)^([0-9a-f]{32})\s+\*?" + re.escape(archive_name) + r"\s*$", checksum.read_text())
    if len(matches) != 1 or hash_file(archive, "md5").lower() != matches[0].lower():
        raise SystemExit("InterProScan official MD5 check failed; archive retained for diagnosis")
    ipr_dir.mkdir(exist_ok=True)
    command_path = ipr_dir / f"interproscan-{args.interpro_version}" / "interproscan.sh"
    ipr_marker = root / "interpro_ready.json"
    archive_sha = hash_file(archive)
    if not ipr_marker.exists() or not command_path.is_file():
        extract_checked(archive, ipr_dir)
    elif read_json(ipr_marker).get("archive_sha256") != archive_sha:
        raise ValueError("InterPro archive changed since extraction")
    if not command_path.is_file():
        raise SystemExit("InterProScan executable not found after extraction")
    write_json(ipr_marker, {"archive_sha256": archive_sha, "interproscan": str(command_path)})
    resources = {**plan, "prepared_utc": utc(), "blast_db": str(db / args.database),
                 "interproscan": str(command_path), "interpro_archive_sha256": archive_sha,
                 "blast_database_info": info, "pfam_db": None}
    # Record file inventory and supplied BLAST checksums; never invent a historical release.
    resources["blast_files"] = [{"name": f.name, "bytes": f.stat().st_size}
                                for f in sorted(db.iterdir()) if f.is_file()]
    resources["blast_md5_files"] = {f.name: f.read_text() for f in db.glob("*.md5")}
    if args.with_pfam:
        pfam_archive = downloads / f"Pfam-{args.pfam_version}-A.hmm.gz"
        download(pfam_url, pfam_archive)
        pfam_dir = root / "pfam"
        pfam_dir.mkdir(exist_ok=True)
        hmm = pfam_dir / "Pfam-A.hmm"
        with gzip.open(pfam_archive, "rb") as source, hmm.open("wb") as target:
            shutil.copyfileobj(source, target)
        command(["hmmpress", "-f", hmm])
        resources.update(pfam_db=str(hmm), pfam_sha256=hash_file(hmm))
    for tool, option in [("blastp", "-version"), ("hmmscan", "-h"), ("java", "-version")]:
        result = subprocess.run([tool, option], capture_output=True, text=True, check=True)
        (root / f"{tool}_version.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    write_json(manifest_path, resources)
    print("LOCAL_RESOURCES_READY", manifest_path)


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.execute:
        return prepare(args)
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("Local resources require Linux x86_64")
    try:
        with run_lock(args.root.expanduser()):
            return prepare(args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
