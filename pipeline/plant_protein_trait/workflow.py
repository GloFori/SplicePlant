"""One entry point for local, online and archived-evidence execution."""
from __future__ import annotations

import copy
import os
from pathlib import Path

from . import annotation, inference
from .core import fasta, fresh_dir, read_json, run_lock, utc, verify_manifest, write_json
from .report import summarize
from .configuration import configure_resources


def parser():
    p = annotation.parser()
    p.description = __doc__
    p.add_argument("--mode", choices=["local", "online", "replay"], required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--forbidden", action="append", default=[])
    p.add_argument("--prepare-only", action="store_true", help="Stop after evidence/prompt preparation without any model charge")
    p.add_argument("--backend", choices=["dashscope", "compatible", "ollama"], default="dashscope")
    p.add_argument("--model", help="Actual provider model ID; required unless --prepare-only")
    p.add_argument("--base-url")
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--num-ctx", type=int, default=32768)
    p.add_argument("--timeout", type=int, default=300, help="Model request timeout")
    p.add_argument("--max-prompt-bytes", type=int, default=120000)
    return p


def execute(args):
    with run_lock(args.output):
        return _execute(args)


def _execute(args):
    configure_resources(args)
    if not args.trait.strip() or (not args.prepare_only and not args.model):
        raise ValueError("Supply a target trait and an actual model ID (or use --prepare-only)")
    if not args.prepare_only:
        key = {"dashscope": "DASHSCOPE_API_KEY", "compatible": "OPENAI_API_KEY"}.get(args.backend)
        if key and not os.environ.get(key):
            raise ValueError(f"Configure {key} before starting, or use --prepare-only")
        if args.backend == "compatible" and not (args.base_url or os.environ.get("OPENAI_BASE_URL")):
            raise ValueError("Configure OPENAI_BASE_URL or --base-url")
    if args.mode == "local":
        if not args.blast_db or not args.interproscan:
            raise ValueError("Local mode requires --blast-db and --interproscan; see references/local.md")
        if any([args.online_blast, args.interpro_remote, args.blast_xml, args.interpro_json,
                args.blast_rid, args.interpro_job_id, args.afdb_accession]):
            raise ValueError("Local mode must not silently use remote annotation or alternate core evidence")
    elif args.mode == "online":
        if any([args.blast_db, args.interproscan, args.blast_xml, args.interpro_json, args.pfam_db]):
            raise ValueError("Online mode uses remote BLAST/InterPro; use --pfam-domtbl only for a declared shared supplement")
        if not args.email:
            raise ValueError("Online mode requires your contact --email")
        args.online_blast = args.interpro_remote = True
    else:
        if not args.blast_xml or not args.interpro_json:
            raise ValueError("Replay requires --blast-xml and --interpro-json")
        if any([args.blast_db, args.interproscan, args.online_blast, args.interpro_remote,
                args.blast_rid, args.interpro_job_id, args.pfam_db, args.afdb_accession]):
            raise ValueError("Replay imports archived evidence without annotation computation")
    query = fasta(args.fasta)
    spec = {"sequence_sha256": query["sequence_sha256"], "sequence_id": query["id"], "mode": args.mode,
            "trait": args.trait, "forbidden": args.forbidden, "backend": args.backend, "model": args.model,
            "base_url": args.base_url, "max_tokens": args.max_tokens, "num_ctx": args.num_ctx,
            "max_prompt_bytes": args.max_prompt_bytes}
    root = Path(args.output).resolve()
    if args.resume:
        previous = read_json(root / "workflow_request.json")
        if previous != spec:
            model_keys = {"backend", "model", "base_url", "max_tokens", "num_ctx"}
            same_inputs = {k: v for k, v in previous.items() if k not in model_keys} == {k: v for k, v in spec.items() if k not in model_keys}
            attempted = any((root / "inference").glob("*/run_status.json"))
            if not same_inputs or attempted:
                raise ValueError("Workflow settings changed after an attempt; choose a new output directory")
            write_json(root / "workflow_request.json", spec)
    else:
        fresh_dir(root)
        write_json(root / "workflow_request.json", spec)
    try:
        ann_args = copy.copy(args)
        ann_args.output = str(root / "annotation")
        ann_args.resume = args.resume and (root / "annotation/request.json").exists()
        ann = annotation.collect(ann_args)
        prep = copy.copy(args)
        prep.annotation_dir = str(ann)
        prep.out = str(root / "inference")
        prepared = root / "inference"
        if not prepared.exists():
            inference.prepare(prep)
        else:
            verify_manifest(prepared, read_json(prepared / "inputs_frozen.json"))
        if args.prepare_only:
            write_json(root / "workflow_status.json", {"status": "prepared_not_executed", "updated_utc": utc()})
            return root
        model = copy.copy(args)
        model.prepared, model.analysis, model.mode = str(prepared), "both", "both"
        if not (prepared / "results_frozen.json").exists():
            inference.run(model)
        inference.freeze(prepared)
        comparison = root / "comparison"
        if not comparison.exists():
            summarize(prepared, comparison)
        else:
            verify_manifest(comparison, read_json(comparison / "report_manifest.json"))
        write_json(root / "workflow_status.json", {"status": "completed", "updated_utc": utc(),
                   "scientific_review": "required"})
        return root
    except Exception as exc:
        write_json(root / "workflow_status.json", {"status": "pending" if isinstance(exc, annotation.PendingJob) else "failed",
                   "error": str(exc), "updated_utc": utc()})
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(execute(args))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from None
