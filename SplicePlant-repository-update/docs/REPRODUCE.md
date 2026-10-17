# Reviewer reproduction workflow

Run shell commands from the root of the maintained SplicePlant checkout. Keep the exact Git commit, downloaded checksums, input sequences and output directories with each experiment.

## 1. Install the DNA-model environment

```bash
conda create -n spliceplant-dna python=3.10 pip -y
conda activate spliceplant-dna

# Choose ONE PyTorch build.
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
# For a compatible NVIDIA CUDA 12.1 installation, use this instead:
# python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
mkdir -p runs/provenance
git rev-parse HEAD > runs/provenance/git_commit.txt
python -m pip freeze > runs/provenance/python_packages.txt
```

The PyTorch commands follow the [official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/). GPU compatibility depends on the server driver and hardware. The requirements retain core historical versions and add dependencies directly used by the scripts; a clean Linux installation of this environment remains a reviewer/server check. Do not substitute the separate protein-to-trait environment automatically.

## 2. Download and verify released assets

```bash
python data/download_assets.py --file model.zip --extract
TRAINING_ASSETS="$PWD/assets/zenodo_18783953/model_extracted/model/training"
WEIGHTS="$TRAINING_ASSETS/saved_model/student_best.pth"
sha256sum "$WEIGHTS"
```

Expected checkpoint SHA-256:

```text
733fc94233480d05b2371d592907a6cb37c42089a0c5c4c9817b4657f5981bbf
```

The helper verifies the full archive's published MD5 before extraction. The 515 MB model archive includes code, a training CSV, the teacher archive and the student checkpoint. Student inference needs only the checkpoint after extraction; it does not need BLAST, InterProScan, an API key or a teacher model.

The helper can be rerun to resume an incomplete transfer. If extraction already completed, reuse that directory and omit `--extract`. A checksum failure or conflicting destination is reported instead of overwriting existing data.

## 3. Check the software

```bash
export SPLICEPLANT_TEST_WEIGHTS="$WEIGHTS"
python -m unittest discover -s tests -v

python inference/score_sequence_per_base.py \
  --fasta data/demo/toy_dna.fa --ckpt "$WEIGHTS" --device cpu \
  --out runs/toy/scores.tsv --call-prefix runs/toy/sites
head runs/toy/scores.tsv
```

The table should have one header and 41 data rows, with finite class probabilities summing approximately to one. Candidate-call files may contain only their header: this is a synthetic sequence, and a candidate splice site is not required for the check to succeed. The tests save temporary files under `runs/software_tests/` and use synthetic weights except for the optional released-checkpoint test.

These checks establish software operation, not biological accuracy.

## 4. Analyze DNA sequences

For one oriented genomic sequence, use [per-base scoring](../inference/README.md#per-base-probabilities). Optional substitution or deletion arguments add mutant probabilities and differences in WT-relative coordinates.

For a directory of sequences that will be explained by attribution:

```bash
python inference/predict_labels.py \
  --input_dir /path/to/fasta_directory --output_dir runs/labels \
  --weights "$WEIGHTS" --device cuda --strand as-is

python attribution/compute_contributions.py \
  --input_dir runs/labels --output_dir runs/attribution --weights "$WEIGHTS" \
  --device cuda --batch_size 32 --num_refs 3 --seed 42 --no_tfmodisco
```

Use CPU where needed. See the inference guide for the differences between complete-window label prediction and zero-padded per-base scoring. Consult the attribution guide before interpreting historical contribution or hypothetical-score columns. The current attribution command extracts candidate regions; it does not run TF-MoDISco clustering.

## 5. Evaluate or retrain

Evaluate the released model on an independently defined labeled CSV:

```bash
python inference/predict.py --csv /path/to/independent_test.csv \
  --weights "$WEIGHTS" --device cuda --log_dir runs/independent_evaluation
```

For retraining, follow [training/README.md](../training/README.md) to unpack the local teacher and run the existing distillation procedure. The training script now saves its row split and configuration beside the checkpoint. Its internal validation set is used for early stopping and is not an independent test set.

To recreate a particular manuscript result, match the released model, evaluation sequences, split, strand/coordinate convention, thresholds and figure processing. The model archive provides substantive inputs, but the metadata inspected here do not establish every experiment-specific evaluation split or figure command. Record those mappings alongside any manuscript-level reproduction.

## 6. Inspect historical attribution and protein examples

```bash
python data/download_assets.py --file moif_results.zip --extract
```

This retrieves the author-archived attribution outputs. It does not claim the current run regenerated them.

The [Arabidopsis FLM example](../pipeline/examples/arabidopsis_flm/README.md) is an independent protein-function example with historical outputs and an annotation replay. The [pipeline guide](../pipeline/README.md) describes its manual workflow and optional AI Skill. A DNA prediction must first be converted into a biologically justified transcript/CDS/protein before entering that workflow; that reconstruction is not performed automatically by the scripts here.
