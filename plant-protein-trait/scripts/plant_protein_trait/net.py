"""Bounded HTTP calls with raw response snapshots; never logs authentication headers."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .core import digest, utc, write_json


def request(url: str, *, form=None, payload=None, headers=None, timeout=60) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.query and "key=" in parsed.query.lower():
        raise ValueError("Credentials must not be embedded in URLs")
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise ValueError("Use HTTPS, or HTTP on localhost only (use an SSH tunnel for remote Ollama)")
    data = None
    h = {"User-Agent": "plant-protein-trait/0.1.0", **(headers or {})}
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        h["Content-Type"] = "application/x-www-form-urlencoded"
    if payload is not None:
        data = json.dumps(payload, allow_nan=False).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        # Provider error bodies can contain private request details; do not copy them to logs.
        raise RuntimeError(f"HTTP {exc.code} from {parsed.hostname}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Network request failed for {parsed.hostname}: {type(exc).__name__}") from None


def snapshot(directory: Path, name: str, body: bytes, source: str, **metadata) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    write_json(path.with_name(path.name + ".provenance.json"), {
        "source": source, "retrieved_utc": utc(), "sha256": digest(body), **metadata})
    return path
