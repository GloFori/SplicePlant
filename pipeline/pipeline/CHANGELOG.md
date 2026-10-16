# 0.3.0

- Consolidate the public workflow under the existing `pipeline/` directory with seven public scripts.
- Separate human instructions from the optional `skill/` agent package; both call the same engine.
- Add dedicated environment installation, chained database setup, installed Skill path binding and manifest-driven resource configuration.
- Preserve completed BLAST downloads across later installation failures; verify recorded resources before reuse.
- Restore optional AFDB retrieval, CIF conversion and exact full-chain mapping, with sequence-bound PDB evidence and declared origin.
- Keep upgraded single-protein inference under the historical public filename `llm.py`.
- Retain original scientific limitations, independent model calls and frozen provenance. No biological demo or live-run success is implied.

# Changelog

## 0.2.0 — 2026-10-15

- Add one-command local/online/replay execution through the same annotation, four-arm inference and reporting code.
- Restore the original nr/five-hit defaults; explicitly align local and remote BLAST settings. Recover v0.1 Swiss-Prot jobs with their original explicit parameters.
- Execute local InterProScan with GO/pathways and local computation; retain optional standalone HMMER/Pfam and allow its raw evidence to be imported in either route.
- Add pinned local software environment, database/InterProScan download preparation, checksum verification and safe extraction.
- Submit independent online annotation stages concurrently; persist jobs, polling states, verified caches and bounded recovery without automatic repeated submissions.
- Add `--resume`, old task-ID recovery, clear network error categories and run locks; skip verified completed model calls without repeating paid requests.
- Add normalized-evidence comparison and deterministic evidence ordering across routes; keep live database/model variability explicit.
- Reject DNA-only inputs; retain existing single-protein scientific evidence rules.
- Extend tests for route equivalence, concurrency, network interruption, pending recovery, uncertain submission, setup extraction and complete replay.
- Recognize the author's repository-root `plant-protein-trait/` folder in CI; document both execution routes in English and Chinese.

## 0.1.0 — 2026-10-14

- Package the maintained single-protein workflow as a self-contained Skill and optional installable Python CLI.
- Replace inherited comparison prompts with independent targeted/blind verify/explore requests; reject multi-record input.
- Read actual legacy `annotation_full.json` dictionary/list formats without inventing final reports.
- Add strict FASTA QC, aligned-query binding for BLAST and full-sequence binding for InterProScan.
- Preserve accessions, organisms, best-HSP identity/coverage/E-value/bit score, HSPs and union coverage.
- Extract nested InterPro GO annotations; label all upstream annotations as computational evidence.
- Remove proxy structural/aggregate confidence scoring and broad unsupported isoform mechanism inference.
- Permit explicit local/imported/online evidence sources; online access requires opt-in and a real contact email.
- Retrieve AFDB target structure only after exact full-length sequence matching; record missingness explicitly.
- Save raw responses, exact messages, provider metadata and SHA256 manifests; prevent overwriting attempts.
- Add result schema checks, unknown-evidence rejection and post-freeze background integration.
- Add cross-platform offline tests, synthetic smoke test, CI workflow, bilingual documentation and source audit.
- Leave the real Arabidopsis example to the maintainer; no private cotton case data or credentials distributed.

This release intentionally changes the historical CLI and narrows the supported task to a single protein. See `docs/AUDIT.md` for migration and excluded features.
