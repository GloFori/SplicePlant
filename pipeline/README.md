# SplicePlant protein-to-trait pipeline

This directory provides a human-operated workflow and an optional AI Skill using the **same analysis code**. Supply one protein FASTA and a target trait. The pipeline collects sequence/domain evidence and produces four independent model analyses: blind verify/explore and targeted verify/explore, followed by a comparison report.

Start with the manual workflow below. AI assistance is optional and described afterward. Current release: **0.3.0**. This release supports a single protein; genomic sequence reconstruction, splice prediction and isoform comparisons are outside its scope.

## Seven public scripts

| Script | Purpose | When needed |
| --- | --- | --- |
| `0_install_environment.py` | Create an isolated Conda environment; optionally invoke the database installer | Once per installation |
| `0_download_databases.py` | Download/verify BLAST data and InterProScan; optional standalone Pfam | Once per database snapshot |
| `1_afdb_seq_search_download.py` | Retrieve existing AlphaFold DB structures by accession or Swiss-Prot hits | Optional structure evidence |
| `2_af3_cif_to_pdb.py` | Convert existing CIF/mmCIF files, gzip or ZIP inputs to PDB | Optional format conversion |
| `3_map_af3_pdbs.py` | Extract chains matching the entire query protein exactly | Optional structure mapping |
| `4_cotton.py` | Collect BLAST, InterProScan, optional Pfam and verified structure evidence | Annotation stage |
| `llm.py` | Run the complete workflow, or QC/prepare/infer/freeze/summarize/compare stages | Complete run or individual stages |

The seven files import `plant_protein_trait/`; **keep the whole directory together**. Scripts 1–3 are optional branches, not three mandatory steps before every analysis. They do not perform de novo AlphaFold prediction. The public filename `llm.py` now exposes the upgraded `new_llm.py` analysis lineage, not the older comparison implementation. See [audit and migration](docs/AUDIT.md).

## Manual workflow

### 1. Obtain the code

On a Linux x86_64 server with Conda and Python available:

```bash
git clone --branch pre-pipline --single-branch https://github.com/GloFori/SplicePlant.git
cd SplicePlant/pipeline
PPT_PACKAGE="$PWD"
PPT_INSTALL="$HOME/ppt_install_v030"
```

The branch must contain this release. For a manuscript reproduction, check out the exact published commit and retain it with the results. Alternatively extract the release ZIP and enter its `pipeline` directory. No GitHub access is needed for the scripts once downloaded.

### 2. Install software and databases once

```bash
python 0_install_environment.py --root "$PPT_INSTALL" --with-databases --execute
conda activate "$PPT_INSTALL/env"
python llm.py doctor
```

The installer creates a dedicated environment, downloads the original core annotation resources (NCBI **nr** and InterProScan 5.78-109.0), verifies downloads/readability and writes `local_config.json`. First installation is a large download and needs substantial disk space, including archives and extracted files. Use your cluster's approved storage and compute/download nodes. A shared filesystem's free space is not your personal quota. Existing Conda is required; the installer does not replace your base environment.

Omit `--execute` to inspect the plan. Repeat the identical command after an interruption; completed resources are checked and reused. Preserve the installation directory for later proteins. Do not change database/releases in an existing installation. Standalone Pfam is optional (`--with-pfam` at installation, `--use-pfam` at analysis). Full setup details, separate installer commands and reuse of v0.2 resources are in [the manual](references/manual.md).

### 3. Check the software without a real protein or API key

```bash
python -m unittest discover -s tests -v
PPT_SMOKE="$(mktemp -d "$HOME/ppt_smoke_XXXXXX")"
python llm.py smoke-test --output "$PPT_SMOKE/check"
```

These are synthetic software checks, not biological validation or a live database/model run. No real Arabidopsis demonstration is bundled yet; see [example requirements](examples/README.md).

### 4. Run one protein

Set your actual paths, trait and provider model identifier:

```bash
PPT_FASTA='/absolute/path/to/your.protein.fa'
PPT_TRAIT='your target trait'
MODEL_ID='your actual DashScope model ID'
PPT_RUN="$HOME/ppt_runs/case_001"
python llm.py qc --fasta "$PPT_FASTA"
```

In the terminal, run these lines and enter your API key when prompted. Do not put the key in the FASTA, GitHub or AI chat:

```bash
read -r -s -p 'DashScope API key: ' DASHSCOPE_API_KEY
printf '\n'
export DASHSCOPE_API_KEY
```

Then run:

```bash
python llm.py pipeline --mode local \
  --config "$PPT_INSTALL/local_config.json" \
  --fasta "$PPT_FASTA" --trait "$PPT_TRAIT" \
  --threads 4 --backend dashscope --model "$MODEL_ID" \
  --out "$PPT_RUN"
```

This runs annotation → evidence/prompt preparation → four isolated model calls → freezing → comparison. `--threads` must fit your allocated resources. Local annotation avoids public annotation queues; the hosted model still needs network access and your provider credentials. Other supported backends are described in [model settings](references/workflow.md).

To run annotation and prepare prompts without model calls, add `--prepare-only` and omit `--model` if unknown. Later repeat with `--resume`, the model and credentials, and without `--prepare-only`. A completed full run has `workflow_status.json` status `completed`. Pending/failed states are not completion.

### 5. Inspect and archive results

| Output | Contents |
| --- | --- |
| `annotation/annotation_full.json` and `annotation_summary.tsv` | Parsed protein evidence |
| `annotation/raw/`, `observed_versions.json`, `evidence_manifest.json` | Original results, commands, versions and hashes |
| `inference/blind_verify`, `blind_explore`, `targeted_verify`, `targeted_explore` | Separate messages, raw responses and validated JSON |
| `inference/inputs_frozen.json`, `results_frozen.json` | Evidence/prompt/result integrity records |
| `comparison/comparison.md`, `evidence_table.tsv` | Targeted-versus-blind comparison, ranked hypotheses and evidence table |
| `workflow_request.json`, `workflow_status.json` | Requested settings and actual completion state |

Archive the whole run directory, input FASTA, code commit, installation metadata and database snapshot. Model conclusions require scientific review; function compatibility is not proof of phenotype causality. Detailed [manual](references/manual.md), [optional structures](references/structures.md), [online route](references/online.md), [reproducibility](references/reproducibility.md) and [validation record](docs/VALIDATION.md) are provided.

## Optional AI Skill

The Skill is in [`skill/SKILL.md`](skill/SKILL.md). It helps an AI with **terminal and file access on the same server** operate this manual workflow. Reading a Markdown file in a chat without execution tools cannot install software or run analysis.

Two usage options:

1. Without installation, ask the server's AI agent to read the absolute path to `pipeline/skill/SKILL.md` and follow it.
2. If your AI host supports installed skills, copy it to that host's documented skill directory using the installer below. The host-specific directory is deliberately not assumed.

```bash
python skill/scripts/ppt.py install \
  --destination /absolute/path/to/your-ai-skills/plant-protein-trait
```

This installs only the Skill and records the existing pipeline's absolute location. Keep the pipeline checkout accessible; the Skill does not carry a second copy of the engine. Restart/reload skills as required by your AI host. See [Skill installation](skill/README.md).

Example request to the AI (replace all paths/model/trait):

> Read `/path/to/SplicePlant/pipeline/skill/SKILL.md`. Use the local route to analyze `/path/to/protein.fa` for the trait `my target trait`. Set up software and databases under `/path/to/ppt_install` if needed. Use DashScope model `my available model ID`; the key is already in the server environment. Write results to `/path/to/run_001`. Run the complete workflow and report the actual status and result paths.

The Skill delegates setup to `0_install_environment.py` and analysis to `llm.py pipeline`. It does not use its own conversation as a replacement for the four independent model calls. Humans and AI use the same resource settings, input validation and result files.
