# Repository maintenance and verification

This update was prepared against `GloFori/SplicePlant`, branch `pre-pipline`, commit `1631e9d970565ffd7f23d47895339417f5721e0c`. It updates the DNA-model entry points and documentation. It contains no replacement files for `pipeline/`.

## Layout correction on 2026-10-17

The maintained scripts stay at their original paths under `training/`, `inference/`
and `attribution/`. Shared model/input helpers and tests now live under `inference/`;
Zenodo metadata live under `data/`. The root `README.md` is the overview and
reviewer entry point. No new top-level package, docs or tests directory is required.
The analysis calculation is unchanged by this layout correction.

The corrected layout was checked in a separate temporary checkout on 2026-10-17:
all ten software tests passed (including the released checkpoint), all seven
command-line entry points displayed help successfully, and all 24 local Markdown
links resolved. The shared model file is byte-identical to the previously checked
implementation. Tests and bytecode caches are excluded from the upload staging
directory except for the source test file itself. The existing `pipeline/` tree
was checked against the prior repository snapshot and was unchanged.

## What changed

- Added the root workflow guide, module guides, reviewer workflow and a verified Zenodo asset downloader.
- Reused the original student model in `inference/spliceplant_model.py`, retaining tensor names, dimensions, convolutions and forward calculation. All inference and attribution entry points now load checkpoints strictly and select the architecture from their parameters. Training chooses attention explicitly.
- Replaced hard-coded dataset paths with command-line arguments, seeded the existing sampling procedure and bounded its negative-sampling loop.
- Validated labeled CSV input, required exactly one FASTA record in single-sequence scoring, and checked substitution reference bases.
- Preserved all FASTA records by writing separate two-line files for attribution, keeping source headers in JSON metadata, and refusing to overwrite existing prediction files. The old ignored `--output_file` argument is replaced by required `--output_dir`.
- Corrected deletion progress labeling and retained NaN gaps during candidate calling.
- Fixed training teacher-device selection, creation of the checkpoint parent directory, empty-training-batch detection, and three-class evaluation reporting. Saved run arguments and row splits. Renamed the final internal score from test accuracy to validation accuracy because the same partition is used for early stopping.
- Removed the unused mandatory TF-MoDISco import from the attribution path. Explicit array export is now available; no TF-MoDISco clustering implementation is claimed.

The model forward calculation, teacher 3-mer preprocessing, distillation loss, historical attribution formula, label mapping and original sequence-edge policies were preserved. Dataset regeneration now has an explicit seed, so it should be treated as generating a new dataset. Interpretive limitations of the original attribution formula are documented in [attribution/README.md](../attribution/README.md), rather than silently changing paper outputs.

## Checks actually performed

Validation host: Windows, Python 3.14.4, PyTorch 2.12.0+cpu, NumPy 2.4.6. This is a maintenance test host, not a claim that the Linux/Python 3.10 historical environment was installed or validated here.

- Inspected the public Zenodo record, both published archive names/checksums, the model ZIP central directory and the first seven complete rows of the training CSV.
- Retrieved the released student checkpoint as a ZIP member, verified its member CRC and computed its SHA-256. This member check is not a claim that the entire 515 MB archive was downloaded and MD5-verified on the maintenance host.
- Loaded all 247 checkpoint entries strictly. Confirmed a 401-bp input and three output classes with identity attention.
- Compared the original and shared model forward calculation using the released weights on two seeded windows: outputs were bit-identical on this CPU host (maximum absolute difference 0).
- Passed ten software tests, including actual released-checkpoint inference, wrapped/prefixed checkpoint loading, rejection of incomplete weights and incompatible window lengths, labeled CSV validation, multiple-FASTA rejection in single-sequence scoring, SNP reference checking, deletion coordinate/gap handling, scoring CLI output, multi-record label output, checksum checking and unsafe ZIP-path rejection.
- Compiled the maintained Python files and checked local documentation links against the source tree and the existing repository's pipeline paths.

Not performed: full student retraining, teacher archive execution, a complete clean Linux/CUDA environment installation, regeneration of manuscript-scale attribution/motifs, or validation of manuscript figures and cross-species metrics. The recommended pinned environment and those scientific runs remain server-side validation steps.

## Historical artifacts

The source snapshot and original mixed CUDA environment inventory remain available in the release/Git history. The versioned Zenodo record is [10.5281/zenodo.18783953](https://doi.org/10.5281/zenodo.18783953). Its metadata describe the archive license; this maintenance package does not assign a new repository-wide license or invent a paper citation.
