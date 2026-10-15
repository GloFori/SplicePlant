# Plant Protein to Trait

A self-contained AI skill **and** a standalone Python workflow for single-protein function-to-trait analysis.

**Release 0.1.0 is a tested software workflow, not a validated biological case study.** The Arabidopsis example will be added by the maintainer. No cotton sequences, association results, credentials or private server paths are distributed. Offline tests use synthetic fixtures explicitly marked as such. Real online annotation services and LLM backends require a separate validation run in your environment.

中文使用说明：[README.zh-CN.md](README.zh-CN.md).

## What it does

```text
one protein FASTA → QC → BLAST / InterProScan / Pfam / optional exact-match AFDB
                                     ↓
                           frozen protein evidence
                                     ↓
               ┌─────────────────────┴─────────────────────┐
     targeted verify / explore                   blind verify / explore
       supplied target trait                      no target trait supplied
               └─────────────────────┬─────────────────────┘
                             freeze all four results
                                     ↓
                  A/B comparison → post-freeze study interpretation
```

This release maintains the **`new_llm.py`** analysis lineage; `llm.py` is historical only. The implementation has been refactored for single-protein analysis rather than wrapping the original comparison prompts. The original seven-file audit and migration details are in [docs/AUDIT.md](docs/AUDIT.md).

## Quick check, no account or API key

From this directory, with Python **3.10 or later**:

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py --output runs/smoke
```

No pip packages, model download, database download or network request are needed. Choose a new output directory for repeated smoke tests. PASS verifies software interfaces and prompt preparation, not biological inference. Tests with a mocked provider do not spend API credits and are not scientific results.

Optional command-line installation:

```bash
python -m pip install .
ppt-annotate --help
ppt-infer --help
ppt-summarize --help
```

The direct `python scripts/...` commands remain available without installing a package. A source checkout contains the Skill; the Python wheel contains the command-line engine, not the Skill documentation.

## Install as a Skill

Copy this complete `plant-protein-trait` directory into your skill host's personal skills folder. For Codex, use `$CODEX_HOME/skills/plant-protein-trait` or `~/.codex/skills/plant-protein-trait`. The resulting path must contain `SKILL.md`, `scripts/`, `references/` and `agents/` together. Start a new task after installation.

Example request:

> Use $plant-protein-trait to analyze my single plant protein. Use the provided FASTA and existing annotation results. Run targeted validation for my specified trait and an independent blind analysis. Preserve raw evidence and do not infer causality.

## Run with your protein

Put inputs in a local directory such as `local_data/` (ignored by Git). Replace the paths and trait in these examples with your actual inputs.

```bash
python scripts/check_sequence.py --fasta local_data/protein.fa --output runs/qc.json

# Reuse raw results (fully offline). The parsers check query/sequence binding.
python scripts/4_cotton.py --fasta local_data/protein.fa --output runs/annotation \
  --blast-xml local_data/blast.xml --interpro-json local_data/interproscan.json

# Prepare four independent prompts. This command does NOT call a model.
python scripts/new_llm.py prepare --fasta local_data/protein.fa \
  --annotation-dir runs/annotation --out runs/inference --trait "your target trait"
```

Add `--forbidden "study-specific label"` as often as needed. Read [references/workflow.md](references/workflow.md) for local database and online annotation alternatives, including the required contact email. No output directory is silently overwritten.

Run a configured backend, then freeze and compare:

```bash
# MODEL_ID must identify a model already available from your running Ollama server.
python scripts/new_llm.py run --prepared runs/inference --backend ollama --model "$MODEL_ID"
python scripts/new_llm.py freeze --prepared runs/inference
python scripts/summarize.py --prepared runs/inference --output runs/comparison
```

DashScope and a generic chat-completions-compatible endpoint are also supported. Credentials come only from environment variables. No model is selected or downloaded automatically. All four calls use fresh messages without shared chat history. If no model is configured, stop after prompt preparation and record `prepared_not_executed`.

## Outputs and reproducibility

| Stage | Key artifacts |
| --- | --- |
| QC and annotation | `query.fasta`, `sequence_qc.json`, raw service/tool responses, `annotation_full.json`, BLAST/domain/GO tables, structure status, `evidence_manifest.json` |
| Preparation | Neutral-ID `protein_evidence.json`, four `messages.json` files, `query_mapping.json`, `preparation.json`, `inputs_frozen.json` |
| Inference | Separate provider response, text response, result JSON, Markdown report and status for each call |
| Freeze | `results_frozen.json`, requiring all four completed and valid results |
| Comparison | `comparison.md`, `evidence_table.tsv`, optional post-freeze background and `report_manifest.json` |

Record the repository commit/tag, Python version, raw database responses and their versions, exact model ID, prompt files, temperature and all raw outputs. Live databases and hosted model aliases change; identical commands do not guarantee identical future outputs. Hashes detect changes relative to a manifest; they are not cryptographic signatures or proof that data are true.

## Scientific scope

- Exactly one protein. Multi-isoform comparison, splicing prediction, association analysis and causal testing are outside this release.
- Homology/domain/GO evidence can suggest function. It does not establish that a gene controls a particular trait.
- Optional AFDB retrieval validates the entire sequence before assigning target structural evidence. It does not run de novo AlphaFold, Foldseek or TM-align.
- Missing stages stay missing. There is no aggregate biological confidence score or proxy structure score.
- Model output validation checks format and evidence references; it does not prove the model's interpretation is correct.
- Blind means isolated inference requests without the target or study context, not absence of knowledge in the pretrained model or a guarantee against indirect annotation clues.
- Verify is the conservative interpretation; explore is hypothesis generation. Report them separately.

Read [interpretation rules](references/interpretation.md), [schema details](references/schemas.md), and [validation status](docs/VALIDATION.md).

## Add to an existing GitHub repository

Keep this folder intact, for example at `skills/plant-protein-trait/`. The bundled workflow can locate it in that position or at the repository root. See [docs/GITHUB_HANDOFF.md](docs/GITHUB_HANDOFF.md) for commands, CI placement, releases and citation instructions. This delivery does not create a GitHub repository or publish files.

No new license or author identity has been imposed on your existing repository. Before public reuse, apply the repository's actual license and author/citation metadata; see [NOTICE.md](NOTICE.md).
