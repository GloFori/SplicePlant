# Changelog

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
