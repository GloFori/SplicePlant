"""Bounded HTTP calls with raw response snapshots; never logs authentication headers."""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .core import digest, utc, write_json


class NetworkFailure(RuntimeError):
    """A request failed before a usable HTTP response was received."""


class HTTPFailure(RuntimeError):
    def __init__(self, host, code, retry_after=0):
        super().__init__(f"HTTP {code} from {host}")
        self.code = code
        self.retry_after = retry_after


def request(url: str, *, form=None, payload=None, headers=None, timeout=60) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.query and "key=" in parsed.query.lower():
        raise ValueError("Credentials must not be embedded in URLs")
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise ValueError("Use HTTPS, or HTTP on localhost only (use an SSH tunnel for remote Ollama)")
    data = None
    h = {"User-Agent": "plant-protein-trait/0.3.0", **(headers or {})}
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
        delay = exc.headers.get("Retry-After", "0") if exc.headers else "0"
        raise HTTPFailure(parsed.hostname, exc.code, int(delay) if delay.isdigit() else 0) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        detail = type(reason).__name__
        if isinstance(reason, ssl.SSLCertVerificationError):
            detail += ": TLS certificate verification failed"
        elif isinstance(reason, OSError) and reason.errno is not None:
            detail += f": errno {reason.errno} ({os.strerror(reason.errno)})"
        # Do not print proxy URLs, request bodies, authentication headers or provider error bodies.
        raise NetworkFailure(f"Network request failed for {parsed.hostname}: {detail}") from None


def snapshot(directory: Path, name: str, body: bytes, source: str, **metadata) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    write_json(path.with_name(path.name + ".provenance.json"), {
        "source": source, "retrieved_utc": utc(), "sha256": digest(body), **metadata})
    return path
