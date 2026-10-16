# Detailed manual workflow

All commands below run from the checkout's `pipeline/` directory. Start with [README](../README.md). Linux x86_64 and Conda are prerequisites for the local distribution. Without Conda, use the official [Miniforge installation instructions](https://github.com/conda-forge/miniforge#install). Initial bootstrap requires Python 3.10+; the created analysis environment uses Python 3.12.

## Software and data: combined or separate

```bash
PPT_INSTALL="$HOME/ppt_install_v030"
python 0_install_environment.py --root "$PPT_INSTALL" --with-databases
python 0_install_environment.py --root "$PPT_INSTALL" --with-databases --execute
conda activate "$PPT_INSTALL/env"
```

The first command only prints a plan. The second installs software and calls the database installer. For a long download, run the second command within your site's persistent terminal or permitted batch job and retain its log. Only one installer may operate on a directory at a time.

Alternatively install the two stages separately:

```bash
python 0_install_environment.py --root "$PPT_INSTALL" --execute
conda activate "$PPT_INSTALL/env"
python 0_download_databases.py --root "$PPT_INSTALL/resources" --database nr --execute
```

The Conda specification pins Python 3.12, Biopython 1.85, BLAST+ 2.17.0, HMMER 3.4 and Java 11. Dependency builds are resolved at install time; `environment.explicit.txt` records them. The script does not install a model server, model weights, AlphaFold, Foldseek or TM-align. Those are not required for the supported core analysis.

The database installer downloads a current NCBI preformatted nr snapshot and the fixed InterProScan 5.78-109.0 distribution (including its bundled member databases). It verifies InterPro's official checksum and queries the BLAST database. Data and compressed archives remain outside Git. Budget space for both; do not start this in a small home quota. The generated files are:

```text
INSTALL_ROOT/
  env/
  environment.explicit.txt
  local_config.json
  installation_request.json
  installation_status.json
  resources/
    resources.json
    blast_database_info.txt
    blast_ready.json
    blast/
    interpro/
    downloads/
```

Repeat the identical command to resume. A completed BLAST checkpoint prevents an interrupted later stage from silently refreshing that database. Changed settings, missing cached files or changed inventories stop for diagnosis. A lock left by a killed process must only be removed after confirming that process has stopped. No installer overwrites an unrelated environment.

Optional standalone Pfam: add `--with-pfam` on the first combined setup, or on the separate database command. At analysis add `--use-pfam` with `--config`/`--resources`. This prepares Pfam 38.0 and HMMER indexes; Pfam inside InterProScan is already part of the core route.

Swiss-Prot can be selected with `--database swissprot` during setup if it is the intended study database. That is a different search from nr. The analysis reads the chosen database from the manifest; an explicit conflicting database is rejected. Record the actual snapshot, not just its name.

## Complete analysis

Use the variables and model setup in [README](../README.md), then:

```bash
python llm.py pipeline --mode local \
  --config "$PPT_INSTALL/local_config.json" \
  --fasta "$PPT_FASTA" --trait "$PPT_TRAIT" \
  --threads 4 --backend dashscope --model "$MODEL_ID" --out "$PPT_RUN"
```

Choose a new output directory for each changed input, analysis setting or model attempt. `--prepare-only` stops before any model call. `--resume` checks existing provenance and skips verified completed work. Uncertain/failed model calls are not automatically charged again; retain that attempt and explicitly prepare a new inference directory from the same annotation if retrying.

## Manual execution of the annotation and inference stages

This is an alternative to the preceding complete-run command. Use different output directories:

```bash
PPT_STAGED="$HOME/ppt_runs/staged_case_001"
python llm.py qc --fasta "$PPT_FASTA"
python 4_cotton.py --fasta "$PPT_FASTA" \
  --config "$PPT_INSTALL/local_config.json" --threads 4 \
  --out "$PPT_STAGED/annotation"
python llm.py prepare --fasta "$PPT_FASTA" \
  --annotation-dir "$PPT_STAGED/annotation" \
  --trait "$PPT_TRAIT" --out "$PPT_STAGED/inference"
python llm.py run --prepared "$PPT_STAGED/inference" \
  --backend dashscope --model "$MODEL_ID"
python llm.py freeze --prepared "$PPT_STAGED/inference"
python llm.py summarize --prepared "$PPT_STAGED/inference" \
  --output "$PPT_STAGED/comparison"
```

Run each command only if the preceding command succeeds. `4_cotton.py` alone does not call the model. `llm.py prepare` alone does not call the model either. The `llm.py run` defaults execute all four slots. Only four validated results can be frozen and summarized.

Optional scripts 1–3 and how to attach their outputs are documented in [structures](structures.md). They are not required when no target structure is available.

## Reuse installations and v0.2 resources

Do not delete or download a large completed database again just to update this code. Activate an environment containing the required executables, and supply the existing resource manifest:

```bash
conda activate /absolute/path/to/existing/env
python llm.py pipeline --mode local \
  --resources /absolute/path/to/existing/resources/resources.json \
  --fasta "$PPT_FASTA" --trait "$PPT_TRAIT" \
  --backend dashscope --model "$MODEL_ID" --out "$PPT_RUN"
```

The v0.2 `resources.json` format is accepted. Its recorded database and executable paths must still exist. If its download is currently running, let that installation finish before using the manifest. The earlier environment may lack Biopython; install `biopython=1.85` in a dedicated compatible environment before optional structure conversion/mapping. Core annotation/inference modules use the Python standard library. Save the updated environment export if changed.

Existing resources without our manifest can also be used explicitly:

```bash
python llm.py pipeline --mode local \
  --blast-db /path/to/blast/nr --database nr \
  --interproscan /path/to/interproscan.sh \
  --fasta "$PPT_FASTA" --trait "$PPT_TRAIT" \
  --backend dashscope --model "$MODEL_ID" --out "$PPT_RUN"
```

## Failure and completion checks

Inspect `installation_status.json` for installation failures, `workflow_status.json` for the complete command, and `annotation/run_status.json` for annotation. Staged runs do not have a workflow status file; inspect their stage status and frozen manifests. Pending remote jobs retain their IDs. A successful smoke test validates software wiring only.

The official sources for setup are [NCBI data downloads](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html), [InterProScan downloads](https://interproscan-docs.readthedocs.io/en/v5/HowToDownload.html), [Pfam releases](https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/), and [Bioconda BLAST](https://bioconda.github.io/recipes/blast/README.html). Changes to external resources require revalidation of the relevant installation step.
