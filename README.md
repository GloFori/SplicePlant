<p align="center">
  <img src="./logo.png" alt="SplicePlant logo" width="180">
</p>
# SplicePlant

Large language models reframe cross-species splicing prediction and the interpretation of regulatory mechanisms and functional consequences in plants.

This repository contains the code for preparing sequence data, training the SplicePlant student model, predicting splice sites, analyzing sequence contributions, and interpreting protein function and traits. This page is the overall usage guide; each module has its own detailed instructions.

## Repository structure

```text
SplicePlant/
├── README.md                         # Overall usage guide
├── requirements.txt                  # DNA-model dependencies; install PyTorch separately
├── data/
│   ├── README.md                     # Released data, formats and checksums
│   ├── download_assets.py            # Download and verify the Zenodo archives
│   ├── demo/toy_dna.fa                # Synthetic software-check input
│   └── ...                           # Asset metadata and original Explanation file
├── training/
│   ├── README.md
│   ├── ge_demo_data.py                # Prepare windows from a genome and GFF3
│   ├── kd_splice.py                   # Knowledge-distillation training
│   └── requirements.txt              # Original environment inventory
├── inference/
│   ├── README.md
│   ├── predict.py                    # Evaluate labeled DNA windows
│   ├── predict_labels.py             # Write per-base labels for attribution
│   ├── score_sequence_per_base.py    # Probabilities and variant-effect scores
│   ├── spliceplant_model.py          # Shared student model and checkpoint loading
│   ├── spliceplant_inputs.py         # Shared input validation
│   ├── tests/                        # Software tests
│   └── MAINTENANCE.md                 # Changes and validation record
├── attribution/
│   ├── README.md
│   └── compute_contributions.py      # Contribution scores and candidate motifs
└── pipeline/                         # Protein-to-trait workflow and optional AI Skill
    ├── README.md
    ├── ...                           # Seven entry scripts and shared engine
    ├── examples/arabidopsis_flm/      # Archived FLM example and annotation replay
    └── skill/                        # Optional agent instructions
```

The original five module directories remain the entry points. Keep their relative locations: training, inference and attribution share the two helper files in `inference/`.

## Choose your starting point

| What you want to do | Input | Where to start | Main output |
| --- | --- | --- | --- |
| Use the released model | Oriented DNA FASTA and student weights | [Inference](inference/README.md) | Per-base probabilities and candidate splice sites |
| Assess a substitution or deletion | DNA FASTA, variant specification and weights | [Variant scoring](inference/README.md#single-substitutions-or-motif-deletions) | WT/mutant probabilities and differences |
| Evaluate a labeled dataset | Fixed-length DNA windows in CSV | [Evaluation](inference/README.md#evaluate-labeled-windows) | Accuracy, classification report and AUC when defined |
| Train a model | Training CSV and local teacher checkpoint | [Training](training/README.md) | Student weights, logs and saved split indices |
| Prepare a new training dataset | Matching genome FASTA and GFF3 | [Data preparation](data/README.md#preparing-a-new-dataset) | Labeled sequence-window CSV |
| Explain sequence predictions | DNA/label files and student weights | [Attribution](attribution/README.md) | Contribution scores and candidate motif intervals |
| Analyze protein function and traits | Protein FASTA and a target trait | [Protein-to-trait pipeline](pipeline/README.md) | Annotation evidence and model interpretation reports |
| Inspect the published FLM example | Bundled original inputs and archived outputs | [FLM example](pipeline/examples/arabidopsis_flm/README.md) | Historical results and annotation replay |

The DNA-model workflow and protein workflow have different inputs. DNA predictions do not automatically reconstruct transcripts, CDSs or proteins; supply a biologically justified protein FASTA to the protein workflow.

## Installation and first run

### 1. Obtain the code and install the DNA-model environment

Run the following on Linux. Subsequent DNA-model commands run from the repository root.

```bash
git clone --branch pre-pipline --single-branch https://github.com/GloFori/SplicePlant.git
cd SplicePlant
conda create -n spliceplant-dna python=3.10 pip -y
conda activate spliceplant-dna

# CPU build:
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
# For a compatible NVIDIA CUDA 12.1 server, use this build instead:
# python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt
python -m pip check
```

Choose one PyTorch build according to the server. The commands follow the [official PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/). The root requirements are for DNA training, inference and attribution. `training/requirements.txt` preserves the original environment inventory; the protein workflow has its own installer under `pipeline/`.

### 2. Download the released weights and data

The author-supplied release is [Zenodo record 18783953](https://zenodo.org/records/18783953), DOI [10.5281/zenodo.18783953](https://doi.org/10.5281/zenodo.18783953).

```bash
python data/download_assets.py --file model.zip --extract
TRAINING_ASSETS="$PWD/assets/zenodo_18783953/model_extracted/model/training"
WEIGHTS="$TRAINING_ASSETS/saved_model/student_best.pth"
sha256sum "$WEIGHTS"
```

The approximately 515 MB `model.zip` contains the training CSV, a DNABERT-2 teacher archive, student weights and a historical source snapshot. The helper supports resuming downloads and verifies the published checksum. Reuse completed extraction directories; do not repeat `--extract` into an existing destination.

Expected student-weight SHA-256:

```text
733fc94233480d05b2371d592907a6cb37c42089a0c5c4c9817b4657f5981bbf
```

The released model accepts **401-bp DNA windows** and predicts **acceptor (0), donor (1), neither (2)**. It uses identity attention. The scripts infer window length and architecture from the checkpoint and load all tensors strictly. Student inference does not require the teacher model, BLAST/InterProScan databases, or an API key. Use the maintained repository scripts with the released weights.

### 3. Run a small software check

```bash
export SPLICEPLANT_TEST_WEIGHTS="$WEIGHTS"
python -m unittest discover -s inference/tests -v

python inference/score_sequence_per_base.py \
  --fasta data/demo/toy_dna.fa --ckpt "$WEIGHTS" --device cpu \
  --out runs/toy/scores.tsv --call-prefix runs/toy/sites
head runs/toy/scores.tsv
```

The score table has a header plus 41 rows. Probabilities should be finite and sum approximately to one. The candidate-site files can contain only a header. This synthetic sequence checks software operation; it is not a biological example or a source of manuscript accuracy claims.

## Run each module

### Inference on your DNA

```bash
python inference/score_sequence_per_base.py \
  --fasta /path/to/gene.fa --ckpt "$WEIGHTS" --device cuda \
  --out runs/gene/scores.tsv --call-prefix runs/gene/sites
```

Use `--device cpu` without a GPU. Supply one FASTA record in the desired strand orientation. Outputs contain sequence-relative coordinates and three class probabilities. Optional `--edit '201:G>A'` or `--del '201:TAGAA'` arguments score a substitution or deletion after checking the reference sequence. See [input, coordinate and edge conventions](inference/README.md).

For labeled evaluation, use an independently defined CSV with `seq,label` columns:

```bash
python inference/predict.py --csv /path/to/independent_test.csv \
  --weights "$WEIGHTS" --device cuda --log_dir runs/evaluation
```

### Attribution

First generate labels, then explain the selected windows:

```bash
python inference/predict_labels.py \
  --input_dir /path/to/fasta_directory --output_dir runs/labels \
  --weights "$WEIGHTS" --device cuda --strand as-is

python attribution/compute_contributions.py \
  --input_dir runs/labels --output_dir runs/attribution --weights "$WEIGHTS" \
  --device cuda --batch_size 32 --num_refs 3 --seed 42 --no_tfmodisco
```

Each FASTA record gets its own DNA/label file and source-header metadata. Attribution produces contribution scores and candidate motif intervals. The historical attribution formula is retained; its interpretation limits are explained in [the attribution guide](attribution/README.md). This script does not run TF-MoDISco clustering.

The original attribution results can be downloaded separately:

```bash
python data/download_assets.py --file moif_results.zip --extract
```

`moif_results.zip` is the actual published filename. These are archived author outputs, not results generated by the command above.

### Training and new data preparation

Training uses the released `cotton_data.csv` and the local teacher inside `dnabert2_117M.zip`. Follow [the training guide](training/README.md) to unpack the teacher and set its checkpoint path. The training command saves best student weights, a log, arguments, an input hash and exact training/validation row indices.

To prepare new windows from your own genome:

```bash
python training/ge_demo_data.py \
  --fasta /path/to/genome.fa --gff /path/to/genes.gff3 \
  --out runs/new_data/windows.csv --flank 200 --seed 42
```

The default flank produces 401-bp windows using the original exonic boundary convention. This generates a new sampled dataset; use [the data guide](data/README.md) to check annotation requirements and split design.

### Protein-to-trait analysis and the optional Skill

Start at [pipeline/README.md](pipeline/README.md). Its seven entry scripts cover environment/database setup, optional structure processing, protein annotation and model interpretation. The local route installs annotation resources once; the online route uses public services. Model calls require the user's chosen provider and credentials.

The optional AI interface is [pipeline/skill/SKILL.md](pipeline/skill/SKILL.md). An AI agent with terminal and file access on the server can read it and operate the same workflow; [installation instructions](pipeline/skill/README.md) are provided. The manual route does not require an AI host.

For a bundled real example, see the [Arabidopsis FLM inputs, archived manuscript outputs and annotation replay](pipeline/examples/arabidopsis_flm/README.md). Follow that example's instructions for an offline replay without new database searches or model calls.

## Reproduction records

Save the exact code commit and environment with each run:

```bash
mkdir -p runs/provenance
git rev-parse HEAD > runs/provenance/git_commit.txt
python -m pip freeze > runs/provenance/python_packages.txt
```

For a manuscript-level reproduction, also retain input hashes, the checkpoint, the evaluation split, strand/coordinate conventions, thresholds and figure-processing commands. The internal 80/20 training split is used for early stopping, so its final accuracy is a validation score rather than an independent test result. The released CSV has sequence/label columns but no locus or split column; full-dataset accuracy is not held-out performance.

See [maintenance and validation](inference/MAINTENANCE.md) for the code changes and checks actually performed, and [data availability](data/README.md) for archive checksums. The current checks establish checkpoint compatibility and small-scale CPU inference; they do not claim full retraining, regenerated manuscript motifs or a validated clean Linux/CUDA installation.
