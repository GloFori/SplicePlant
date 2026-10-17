# Attribution and candidate motifs

Use the sequence-model environment from the root README. Captum is included in the root requirements; the protein-to-trait environment is separate.

First generate two-line DNA/label files with [predict_labels.py](../inference/README.md#predict-labels-for-attribution). Then run:

```bash
WEIGHTS="$PWD/assets/zenodo_18783953/model_extracted/model/training/saved_model/student_best.pth"
python attribution/compute_contributions.py \
  --input_dir runs/labels --output_dir runs/attribution --weights "$WEIGHTS" \
  --device cuda --batch_size 32 --num_refs 3 --seed 42 --no_tfmodisco
```

Use CPU if necessary. Attribution memory grows with batch size, reference count and window length; lower the batch size when memory is limited. The model window is inferred from the weights. Checkpoint loading is strict, so missing or unexpected model tensors cause an error.

Input files must each contain exactly two lines: DNA and a label digit per base. Label 0 selects acceptor-centered windows and label 1 selects donor-centered windows. Complete windows are explained against generated first-order Markov reference sequences; label 2 does not select a window. Candidate motifs are contiguous stretches above the absolute-score threshold (default 0.12), at least 5 bp long.

Outputs are `<input_stem>_motifs.txt` and `run_config.json`. Motif intervals use inclusive endpoints in both reported 0-based and 1-based notation. Overlapping windows can produce overlapping candidates; the script does not deduplicate or cluster them into regulatory motifs.

## Historical scoring semantics

This maintenance update preserves the published repository's contribution calculation so it does not silently replace the manuscript analysis. Its `DeepLift(model)` call uses Captum's default multiplication by `(input - baseline)`. The script then masks returned contributions by the observed one-hot input and constructs its historical `hypothetical_contrib` values from those already multiplied contributions.

Those historical hypothetical values are **not raw DeepLIFT multipliers or independently validated TF-MoDISco hypothetical scores**. Residual blocks also retain functional ReLU calls, which are not equivalent to registering every nonlinear activation as a Captum-hooked module. A methodological revision needs separate validation against the original paper outputs; it is not claimed by this packaging update. See the [Captum DeepLift API](https://captum.ai/api/deep_lift.html) for the multiplier convention.

## Optional array export

`--export_tfmodisco_inputs` writes per-class NPZ arrays named `one_hot`, `contribution_scores` and `hypothetical_scores`. They retain the historical calculation above. This flag exports arrays only: **the script does not run TF-MoDISco clustering**. Do not combine it with the compatibility flag `--no_tfmodisco`.

The original code imported TF-MoDISco and collected arrays without running or saving a clustering result. The maintained default no longer requires this unused package. Existing manuscript motifs are available in [the released `moif_results.zip`](https://zenodo.org/records/18783953); use those archived results when inspecting the paper rather than claiming this maintenance pass regenerated them.
