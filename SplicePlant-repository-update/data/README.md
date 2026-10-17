# Data and released weights

## Primary release

[Zenodo record 18783953](https://zenodo.org/records/18783953) is the author-supplied model release. Its public metadata and ZIP directory were inspected for this maintenance update.

| Published file | Bytes | Published MD5 |
| --- | ---: | --- |
| `model.zip` | 515006011 | `43268b79b4cfc40dc119a15cdb7f4e62` |
| `moif_results.zip` | 219980786 | `ae15d9aabb51c5cb9a5a559ba8d6c02c` |

The release description calls the second archive `motif_results.zip`; use the actual file name **`moif_results.zip`** in download commands. The download helper uses the published sizes and checksums, resumes interrupted downloads when supported, and verifies the complete archive before extraction.

```bash
# Run from the repository root. No API key is needed.
python data/download_assets.py --list
python data/download_assets.py --file model.zip --extract

# Optional: download and inspect the historical attribution results.
python data/download_assets.py --file moif_results.zip --extract
```

Extraction goes to a dedicated new directory. A completed extraction directory is not overwritten. To reuse an existing installation, omit `--extract` or use its existing files.

Important paths after extracting `model.zip`:

```text
assets/zenodo_18783953/model_extracted/model/
├── training/
│   ├── cotton_data.csv
│   ├── dnabert2_117M.zip
│   ├── saved_model/student_best.pth
│   └── ... historical training code
├── inference/     historical source
├── attribution/   historical source
└── pipeline/      historical source
```

The student checkpoint was extracted separately, its ZIP CRC checked, and its tensors inspected:

- Size: 6,312,043 bytes; 247 state-dict entries.
- SHA-256: `733fc94233480d05b2371d592907a6cb37c42089a0c5c4c9817b4657f5981bbf`.
- `fc1.weight`: `[512, 401]`; output convolution: `[3, 40, 1]`.
- Positional embeddings and normalization are present; no Sinkhorn attention tensors are present.

The inspected training CSV prefix has columns `seq,label` and 401-bp windows. The full CSV, nested teacher archive and motif archive have not been scientifically revalidated during this maintenance pass. The repository's earlier [source-data record 11213271](https://zenodo.org/records/11213271) is retained as a historical reference; its relationship to particular manuscript analyses is not inferred here.

## Input conventions

For labeled-window evaluation or training, use CSV columns `seq,label`. All windows must have the same odd length. Labels are `0=acceptor`, `1=donor`, `2=neither`. DNA is uppercased; IUPAC ambiguity symbols become all-zero vectors in the model. The released checkpoint expects 401 bp.

For per-base scoring, provide one FASTA record in the desired biological orientation. For directory prediction, each FASTA record is written to its own sequence/label file with an accompanying source-header JSON file. See [inference](../inference/README.md) for strand and edge behavior.

`demo/toy_dna.fa` is a synthetic 41-bp input used to exercise zero-padded inference. It is not a training or biological validation dataset.

## Preparing a new dataset

```bash
python training/ge_demo_data.py \
  --fasta /path/to/genome.fa --gff /path/to/genes.gff3 \
  --out runs/new_data/windows.csv --flank 200 --seed 42
```

This retains the original generator's exonic boundary convention: on the positive strand the donor center is the upstream exon end and the acceptor center is the downstream exon start; windows on the negative strand are reverse complemented. This convention must match the evaluation labels.

The generator remains a demonstration utility. It loads the whole genome, expects compatible mRNA/exon GFF3 annotation, and samples background windows as the `neither` class. Background does not mean experimentally validated absence of splicing. Duplicate windows and related transcripts can remain; curate chromosome/gene/species splits independently before making generalization claims. A seed now controls random sampling and a maximum-attempt bound prevents an endless sampling loop. Regenerated data are not a byte-identical replacement for the released CSV.
