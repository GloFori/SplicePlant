# Training

The released student weights can be used directly; retraining is optional. The original environment inventory remains in `training/requirements.txt` for provenance. For a new installation, use the [root installation guide](../README.md), which separates PyTorch installation from the other dependencies and avoids mixing CUDA 11 and CUDA 12 library inventories.

## Released training inputs

After downloading and extracting `model.zip`:

```bash
TRAINING_ASSETS="$PWD/assets/zenodo_18783953/model_extracted/model/training"
python -m zipfile -l "$TRAINING_ASSETS/dnabert2_117M.zip"
```

Extract the nested teacher archive into a separate directory, then set `TEACHER_DIR` to the directory containing its `config.json`, tokenizer files, model weights and custom model code. Check the archive's top-level paths before choosing that directory.

```bash
mkdir -p assets/teacher
python -m zipfile -e "$TRAINING_ASSETS/dnabert2_117M.zip" assets/teacher
find assets/teacher -name config.json
# Set this to the directory printed above that belongs to DNABERT-2.
TEACHER_DIR=/absolute/path/to/the/extracted/dnabert2_117M

python training/kd_splice.py \
  --csv "$TRAINING_ASSETS/cotton_data.csv" --ckpt "$TEACHER_DIR" \
  --save_path runs/training/student_best.pth --log_dir runs/training/logs \
  --attention identity --device cuda --seed 42 \
  --epochs 30 --batch_size 128 --lr 0.0001 \
  --alpha 0.5 --tau 1.0 --center_mask_prob 0.3 --max_len 512
```

The command uses the repository's existing default hyperparameters; these defaults are not an independently verified record of every manuscript experiment. Use the paper's experiment-specific settings when available. Teacher loading is local/offline and uses the custom code supplied with the checkpoint. The nested teacher archive was located but not loaded during the maintenance validation.

## Preserved training method

The script constructs overlapping 3-mer text, encodes it with the teacher tokenizer/model, mean pools the hidden states, fits a multinomial logistic-regression teacher head, and trains the student using the existing cross-entropy/KL mixture. The original tokenization, loss and convolutional model have been retained. Do not silently replace the teacher preprocessing with a different tokenizer protocol when reproducing an existing run.

`--attention identity` matches the released student checkpoint. To train the optional Sinkhorn architecture, use `--attention sinkhorn` with the dependency version appropriate to that experiment; it produces a different parameter set. Merely having a package installed no longer changes the architecture.

The training command saves:

- Best student weights at `--save_path`.
- `<save_path>.run.json`: arguments, training CSV SHA-256, label mapping, software versions and the exact 0-based row indices used for training and validation.
- Timestamped training/evaluation logs.

The 80/20 stratified row split is used for early stopping; the final score is therefore a **validation** score, not an independent test score. The archived CSV contains sequence and label columns rather than locus or split metadata. A random row split does not establish chromosome-, gene- or species-independent evaluation. Use separately defined evaluation data for manuscript claims and preserve the split definition.

For generating new windows from a genome and GFF3, see [data preparation](../data/README.md#preparing-a-new-dataset). A newly generated CSV is a new dataset, not a reconstructed original archive.
