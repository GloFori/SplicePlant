"""Small standard-library primitives; no network or filesystem writes on import."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def read_json(path: str | Path) -> Any:
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fasta(path: str | Path) -> dict:
    """Reject changes to the submitted sequence, including silent stop removal."""
    path = Path(path)
    records = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip()
            if not header:
                raise ValueError("Empty FASTA header")
            records.append({"id": header.split()[0], "header": header, "sequence": ""})
        else:
            if not records:
                raise ValueError("FASTA sequence before header")
            records[-1]["sequence"] += line.upper()
    if len(records) != 1:
        raise ValueError(f"Single-protein mode requires exactly one record; found {len(records)}")
    rec = records[0]
    invalid = sorted(set(rec["sequence"]) - AA)
    if not rec["sequence"] or invalid:
        raise ValueError(f"Empty protein or invalid amino acids (including stops): {invalid}")
    rec.update(length=len(rec["sequence"]), sequence_sha256=digest(rec["sequence"].encode()),
               fasta_sha256=digest(path.read_bytes()))
    return rec


def fresh_dir(path: str | Path) -> Path:
    """Use a fresh directory instead of mixing results or overwriting evidence."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}; choose a new run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (ValueError, TypeError):
        return default


def manifest(directory: Path, names: list[str], **metadata) -> dict:
    return {"created_utc": utc(), "python": platform.python_version(), **metadata,
            "files": {name: digest((directory / name).read_bytes()) for name in sorted(names)}}


def verify_manifest(directory: Path, data: dict) -> None:
    root = directory.resolve()
    for name, expected in data["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid manifest path: {name}")
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Frozen file changed: {name}")
