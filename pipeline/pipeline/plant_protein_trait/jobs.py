"""Durable remote jobs. Submission is never retried after an ambiguous response."""
from __future__ import annotations

import json
import re
import time
import urllib.parse
from pathlib import Path

from .core import digest, read_json, utc, write_json
from .net import HTTPFailure, NetworkFailure, request, snapshot
from .settings import remote_blast_options

BLAST = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
IPR = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"


class PendingJob(RuntimeError):
    pass


def cached(raw, name):
    path = raw / name
    if not path.exists():
        return None
    sidecar = path.with_name(path.name + ".provenance.json")
    if not sidecar.is_file() or read_json(sidecar).get("sha256") != digest(path.read_bytes()):
        raise ValueError(f"Cached result changed or lacks provenance: {name}")
    return path


def run_job(kind, query, raw, settings, email, max_wait, timeout, supplied_id=None):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    name = "blast.xml" if kind == "blast" else "interpro.json"
    state_path = raw / f"{kind}_job.json"
    id_key = "rid" if kind == "blast" else "job_id"
    binding = {"sequence_sha256": query["sequence_sha256"],
               "settings": settings["blast" if kind == "blast" else "interpro"]}
    state = read_json(state_path) if state_path.exists() else {}
    if state and any(state.get(k) != v for k, v in binding.items()):
        raise ValueError(f"{kind}: saved job does not match the sequence/settings")
    if state.get(id_key) and supplied_id and state[id_key] != supplied_id:
        raise ValueError(f"{kind}: supplied ID differs from saved job")
    saved = cached(raw, name)
    if saved:
        if not state.get(id_key):
            raise ValueError(f"{kind}: cached result has no bound job state")
        print(f"{kind}: reuse verified result {saved}", flush=True)
        return saved
    if supplied_id and not state.get(id_key):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", supplied_id):
            raise ValueError("Invalid remote job identifier")
        state = {**binding, id_key: supplied_id, "status": "submitted", "poll_count": 0,
                 "binding_warning": "User-supplied job ID; original search parameters not independently verified"}
        write_json(state_path, state)
    if not state.get(id_key):
        if state:
            raise RuntimeError(f"{kind}: previous submission outcome is uncertain; recover its job ID before resuming. "
                               "No automatic second submission was made.")
        state = {**binding, "status": "submitting", "submitted_utc": utc(), "poll_count": 0}
        write_json(state_path, state)
        print(f"{kind}: submitting one protein", flush=True)
        if kind == "blast":
            url = BLAST
            form = {"CMD": "Put", "PROGRAM": "blastp", "QUERY": ">query_001\n" + query["sequence"],
                    "tool": "plant-protein-trait", "email": email, **remote_blast_options(settings)}
        else:
            url = IPR + "/run"
            form = {"email": email, "title": "single_protein", "sequence": ">query_001\n" + query["sequence"],
                    "goterms": "true", "pathways": "true"}
            if settings["interpro"]["applications"]:
                form["appl"] = ",".join(settings["interpro"]["applications"])
        try:
            body = request(url, form=form, timeout=timeout)
            snapshot(raw, f"{kind}_submission.txt", body, url, **binding)
            if kind == "blast":
                match = re.search(r"RID\s*=\s*([A-Za-z0-9_.-]+)", body.decode())
                job = match.group(1) if match else None
            else:
                job = body.decode().strip()
            if not job or not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
                raise RuntimeError(f"{kind}: no usable job ID; inspect the saved submission response")
        except Exception as exc:
            state.update(status="submission_uncertain", error=str(exc))
            write_json(state_path, state)
            raise
        state.update({id_key: job, "status": "submitted"})
        if kind == "blast":
            state["database"] = settings["blast"]["database"]
        write_json(state_path, state)
    job = state[id_key]
    print(f"{kind}: task {job}; resume uses this same ID", flush=True)
    deadline = time.monotonic() + max_wait
    interval = 60 if kind == "blast" else 15
    # Shared by polling and result GETs; persists across process restarts.
    next_delay = interval
    while True:
        delay = max(next_delay, interval - (time.time() - state.get("last_request_epoch", 0)),
                    state.get("retry_not_before_epoch", 0) - time.time())
        if time.monotonic() + delay > deadline:
            state.update(status="pending", updated_utc=utc())
            write_json(state_path, state)
            raise PendingJob(f"{kind}: task {job} still pending; rerun with --resume (no new submission)")
        print(f"{kind}: {job}, next check in {int(delay)} s", flush=True)
        time.sleep(delay)
        state.update(last_request_epoch=time.time(), poll_count=state.get("poll_count", 0) + 1)
        write_json(state_path, state)
        getting_result = kind == "interpro" and state.get("remote_status") == "FINISHED"
        url = (BLAST + "?" + urllib.parse.urlencode({"CMD": "Get", "RID": job, "FORMAT_TYPE": "XML2_S"})
               if kind == "blast" else IPR + (f"/result/{job}/json" if getting_result else f"/status/{job}"))
        try:
            body = request(url, timeout=timeout)
        except (NetworkFailure, HTTPFailure) as exc:
            if isinstance(exc, HTTPFailure) and exc.code not in {408, 429, 500, 502, 503, 504}:
                state.update(status="failed", error=str(exc))
                write_json(state_path, state)
                raise
            next_delay = max(interval, getattr(exc, "retry_after", 0))
            state.update(status="connection_error", error=str(exc), updated_utc=utc(),
                         retry_not_before_epoch=time.time() + next_delay)
            write_json(state_path, state)
            print(f"{kind}: {exc}; retaining task {job}", flush=True)
            continue
        next_delay = interval
        snapshot(raw, f'{kind}_poll_{state["poll_count"]:04d}.txt', body, url, **binding)
        if kind == "blast":
            done = any(tag in body for tag in (b"<BlastXML2", b"<BlastOutput"))
            match = re.search(rb"Status\s*=\s*(\w+)", body)
            status = "FINISHED" if done else match.group(1).decode() if match else "UNRECOGNIZED"
        else:
            done = getting_result
            status = "FINISHED" if done else body.decode().strip()
        state.update(remote_status=status, updated_utc=utc())
        print(f"{kind}: {job}, {status}", flush=True)
        if done:
            if kind == "blast":
                import xml.etree.ElementTree as ET
                ET.fromstring(body)
            else:
                json.loads(body)
            path = snapshot(raw, name, body, url, **binding, job_id=job)
            state.update(status="completed", result_sha256=digest(body))
            write_json(state_path, state)
            return path
        if status not in {"WAITING", "READY", "PENDING", "RUNNING", "QUEUED", "FINISHED"}:
            state.update(status="failed")
            write_json(state_path, state)
            raise RuntimeError(f"{kind}: task {job} returned {status}; raw response retained")
        state.update(status="pending")
        write_json(state_path, state)
