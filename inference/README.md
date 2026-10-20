# Inference

Run commands from the repository root after the [environment and asset setup](../README.md#installation-and-first-run).

```bash
WEIGHTS="$PWD/assets/zenodo_18783953/model_extracted/model/training/saved_model/student_best.pth"
```

All entry points use the same student architecture and strict checkpoint loading. Plain state dictionaries, `{"state_dict": ...}` wrappers and uniformly prefixed `module.` keys are supported. Window length and identity/Sinkhorn attention are inferred from the weights. The released checkpoint uses 401 bp and identity attention.

## Per-base probabilities

```bash
python inference/score_sequence_per_base.py \
  --fasta /path/to/gene.fa --ckpt "$WEIGHTS" --device cuda \
  --batch-size 128 --out runs/gene/scores.tsv --call-prefix runs/gene/sites
```

Use `--device cpu` without a GPU. This entry point accepts exactly one FASTA record and scores every base using a centered window. It zero pads positions beyond sequence ends; ambiguity symbols are also all-zero vectors. It does **not** reverse complement FASTA headers automatically. Supply the intended strand orientation.

Outputs:

- `scores.tsv`: sequence identifier, `pos0` (0-based), `pos1` (1-based), base and `pA,pD,pN` probabilities.
- `sites.acceptor.tsv` and `sites.donor.tsv`: thresholded and locally suppressed candidate sites, sorted by score. Default settings are probability ≥ 0.8, margin ≥ 0.2, suppression radius 10 bp, at most 50 sites per class.

Coordinates are relative to the input sequence, not genomic coordinates. Scoring is not transcript assembly. Edge padding differs from a fully contextualized genomic window and should be considered when interpreting edge calls.

## Single substitutions or motif deletions

```bash
# Example syntax; position 201 must actually be G in your own input.
python inference/score_sequence_per_base.py \
  --fasta /path/to/gene.fa --ckpt "$WEIGHTS" --edit '201:G>A' \
  --out runs/snp/scores.tsv --call-prefix runs/snp/sites

# Example syntax; positions 201–205 must actually read TAGAA.
python inference/score_sequence_per_base.py \
  --fasta /path/to/gene.fa --ckpt "$WEIGHTS" --del '201:TAGAA' \
  --out runs/deletion/scores.tsv --call-prefix runs/deletion/sites
```

Reference mismatches fail before mutant scoring. The table contains WT probabilities, mutant probabilities and mutant-minus-WT differences. Deletions shorten the mutant sequence, then align its scores back to the WT coordinate system; removed bases are `nan`. Candidate sites are called on the mutant when a variant is requested and remain in WT-aligned coordinates. Removed positions cannot become candidate sites.

## Predict labels for attribution

```bash
python inference/predict_labels.py \
  --input_dir /path/to/fasta_directory --output_dir runs/labels \
  --weights "$WEIGHTS" --device cuda --strand as-is
```

This preserves the historical complete-window strategy: only A/C/G/T windows wholly contained in the input are evaluated. Unscored sequence ends, windows overlapping ambiguity symbols and positions skipped by `--stride` receive the historical placeholder label `2`. **That placeholder is not a measured probability or evidence of no splice site.** A sequence shorter than the model window is skipped.

Each FASTA record gets a separate `<input_filename>.record0001.txt` file. The first line is DNA, the second has one digit per base (`0`, `1`, `2`). Adjacent JSON metadata preserve the original header, input file, strand policy, window and stride. Existing output files are not overwritten. These two-line files feed the attribution script directly.

`--strand as-is` is explicit and recommended for already oriented sequences. `--strand reverse` reverses every sequence; `--strand header` preserves the original heuristic that reverses headers containing markers such as `(-)`, `minus`, `reverse` or `negative`. Do not use the heuristic on a sequence that has already been oriented. Coordinates in the output refer to the emitted sequence.

This script and the probability scorer intentionally retain their different historical padding/ambiguity policies. Their outputs should not be treated as equivalent at sequence edges or around ambiguous bases.

## Evaluate labeled windows

```bash
python inference/predict.py \
  --csv /path/to/independent_test.csv --weights "$WEIGHTS" \
  --device cuda --batch_size 128 --log_dir runs/evaluation
```

Input CSV requires fixed-length `seq` windows and integer `label` values. The command writes an evaluation log with accuracy, macro one-vs-rest AUC when defined, and a three-class classification report. AUC is reported as unavailable if the supplied labels do not support the calculation. Keep the test set independent of training and early stopping.
