# SplicePlant

Large language models reframe cross-species splicing prediction and the interpretation of regulatory mechanisms and functional consequences in plants.

SplicePlant provides DNA sequence classification and splice-site scoring, training by knowledge distillation, attribution analysis, and a separate protein-to-trait interpretation workflow.

## Choose a workflow

| Goal | Entry point | Required input |
| --- | --- | --- |
| Score splice sites or a sequence variant | [Inference guide](inference/README.md) | DNA FASTA and released student weights |
| Evaluate labeled windows | [Inference guide](inference/README.md#evaluate-labeled-windows) | Fixed-length DNA CSV with labels and student weights |
| Train the student model | [Training guide](training/README.md) | Training CSV and DNABERT-2 teacher checkpoint |
| Compute contribution scores and candidate motifs | [Attribution guide](attribution/README.md) | Predicted sequence/label files and student weights |
| Interpret protein function and traits, manually or with the optional AI Skill | [Protein-to-trait pipeline](pipeline/README.md) | Protein FASTA and the resources described there |
| Inspect the archived Arabidopsis FLM example | [FLM example](pipeline/examples/arabidopsis_flm/README.md) | Bundled original inputs and archived outputs |

The DNA model and the protein-to-trait pipeline have different inputs and environments. A DNA splice-site prediction does not automatically reconstruct a transcript, CDS or protein. The protein workflow begins with a supplied protein sequence.

## Released model and data

The author-supplied release is [Zenodo record 18783953](https://zenodo.org/records/18783953), DOI [10.5281/zenodo.18783953](https://doi.org/10.5281/zenodo.18783953).

- `model.zip`: source snapshot, `cotton_data.csv`, `dnabert2_117M.zip`, and `saved_model/student_best.pth`.
- `moif_results.zip`: archived attribution results. The spelling is the actual published filename.

The released student weights accept **401-bp DNA windows** and return **acceptor (0), donor (1), neither (2)** scores. The checkpoint uses identity attention; Sinkhorn is not required for this release. The maintained scripts infer architecture and window length from checkpoint keys and load every tensor strictly.

See [data availability and checksums](data/README.md). Downloads stay outside Git. Use the maintained scripts in this repository with the downloaded weights; the archived source is a historical snapshot.

## Quick start: sequence-model inference

Run from the repository root on Linux. Clone `pre-pipline` while this development branch is being reviewed; record the exact commit with `git rev-parse HEAD` for each experiment.

```bash
git clone --branch pre-pipline --single-branch https://github.com/GloFori/SplicePlant.git
cd SplicePlant
conda create -n spliceplant-dna python=3.10 pip -y
conda activate spliceplant-dna

# CPU installation; see the reproduction guide for CUDA 12.1.
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check

python data/download_assets.py --file model.zip --extract
WEIGHTS="$PWD/assets/zenodo_18783953/model_extracted/model/training/saved_model/student_best.pth"
```

For a small software check using the released weights:

```bash
python inference/score_sequence_per_base.py \
  --fasta data/demo/toy_dna.fa --ckpt "$WEIGHTS" --device cpu \
  --out runs/toy/scores.tsv --call-prefix runs/toy/sites
```

`data/demo/toy_dna.fa` is synthetic and tests file handling and inference. It is not a biological example or a source of manuscript accuracy claims. Replace it with your own oriented genomic DNA FASTA for analysis. The scorer reports positions relative to the supplied sequence.

## Reproduction and validation

Follow [the complete reviewer workflow](docs/REPRODUCE.md) for downloads, input conventions, training, inference and attribution. [The maintenance notes](docs/MAINTENANCE.md) describe code changes and the checks performed for this update.

Software verification, rerunning inference with released weights, retraining, and reproducing a manuscript figure are distinct tasks. A manuscript-level reproduction also needs the exact evaluation set or split, model configuration and figure-processing commands. The currently archived CSV has no genomic coordinates or split column; do not report its full-dataset accuracy as independent held-out performance.

The existing [protein-to-trait documentation](pipeline/README.md) remains the entry point for that workflow and its Skill.
