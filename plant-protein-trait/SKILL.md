---
name: plant-protein-trait
description: Analyze a single plant protein from FASTA through traceable homology, domain, GO and optional structure evidence, then run isolated targeted and blind trait inference. Use for reproducible protein-function-to-trait studies and reviewer handoffs; not for multi-isoform comparison or causal-variant validation.
---

# Plant protein to trait

Use the scripts bundled with this skill. Resolve script paths relative to this SKILL.md; write analysis outputs into the user's project, not this installed skill directory. Python 3.10+ is sufficient for the core workflow. The maintained inference entry point is `scripts/new_llm.py`; `scripts/4_cotton.py` is a historical filename for cross-species annotation.

## Workflow

1. Establish the exact target FASTA and, for paired analysis, the target trait. Exactly one protein is supported. Validate with `scripts/check_sequence.py`. Do not delete stop codons, concatenate nucleotide files, manufacture a second protein or assume a species from a directory name.
2. Collect independent protein evidence before inference. Prefer supplied raw BLAST XML/InterProScan JSON or existing small local databases. Online submission is explicit and needs the user's contact email for NCBI/EBI. Never download a full nr/PDB/Pfam database for a single target by default.
3. Follow [the execution workflow](references/workflow.md) for annotation, four prompt preparations, model execution and freezing. Read [the schemas](references/schemas.md) when importing legacy outputs or diagnosing parse errors. Use real `annotation_full.json` keyed by sequence ID; do not fabricate a `final_report` or rank table.
4. Prepare targeted verify/explore and blind verify/explore from the same frozen protein evidence. The trait is supplied only to targeted prompts. Supply study-specific leakage phrases using repeated `--forbidden` flags. Keep locus, association, genotype and splicing context out of upstream evidence. The command refuses detected leakage; do not silently remove legitimate biology to force a clean result.
5. Execute four separate stateless calls with a configured backend. A current conversation that already knows the target is not a blind evaluator. Never replace a failed provider call with your own response and label it an independent model result. If no backend is configured, deliver the prepared prompts and state that inference was not run.
6. Freeze all four successful results before opening downstream study background. Use `scripts/summarize.py` for comparison and optional post-freeze context. Then perform scientific review using [interpretation rules](references/interpretation.md). Keep all four raw model outputs unchanged; put corrections or criticism in a separate interpretation.

## Evidence rules

- BLAST statistics describe the best HSP; union coverage is separate. A homolog is not automatically an ortholog, interaction partner or experimentally characterized query.
- InterPro/Pfam matches and derived GO terms are computational evidence. Shared annotation provenance is not independent replication. Do not invent substrate specificity from a broad family.
- A missing structure is null, not zero confidence or a proxy score. An AlphaFold accession is target evidence only for an exact full-length sequence match. pLDDT is not a function or trait score. This release does not automate Foldseek/TM-align or de novo structure prediction.
- `verify` and `explore` describe model prompting styles. They are not experimental verification or independent biological experiments. LLM confidence is qualitative and uncalibrated.
- Report insufficient evidence as such. Targeted-only support is functional compatibility, not independent trait prediction. Stronger claims require independently supplied evidence.

## Delivery

Include FASTA/QC, annotation and raw snapshots, evidence manifest, four messages files, provider metadata and responses if executed, input/result hashes, comparison, evidence table, commands and limitations. Separate prepared, failed, completed and frozen states. Do not claim a real example or server/online validation based on synthetic smoke tests.

For adding a real Arabidopsis example later, use [the example contract](examples/README.md). For installation and repository placement, use [README](README.md) and [the GitHub handoff](docs/GITHUB_HANDOFF.md).
