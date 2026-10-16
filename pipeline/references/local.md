# Local route

Follow [the manual](manual.md) for the complete installation and analysis commands. Both human and Skill execution use the same local engine.

The core method is BLAST against the declared database (default nr) plus full InterProScan 5, with optional independent HMMER/Pfam. InterProScan runs with `-dp` to disable its precalculated online lookup. Hosted model inference still requires network access; local annotation alone does not make inference offline.

Installation lives outside the checkout. The installer writes resource paths and version records; subsequent runs reuse these resources via `--config` or `--resources`. If using a previously prepared installation, do not redownload it: see the manual's v0.2 migration section.

The requested parameters and observed versions are recorded. For local versus online comparisons use [reproducibility instructions](reproducibility.md). Database names alone do not establish equality of snapshots.
