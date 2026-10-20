# Arabidopsis FLM: archived manuscript case and annotation replay

## What is included

The original AlphaFold Server structure exports are now available in [`recovered_server_archive/`](recovered_server_archive/README.md), added on 2026-10-20. This supplement contains the WT and ALT native structure models, confidence records, structure job inputs, converted PDB files, and additional historical reports. These are recovered historical outputs; no new AlphaFold or language-model run was performed.

This example contains the complete set of 15 files supplied in the author's `case2.rar`, organized by purpose, plus a tested v0.3.0 annotation replay and its reference outputs. The author confirms that the original model analysis was completed for the manuscript. The supplied model report is included unchanged; no new model analysis was commissioned during packaging.

| Directory / file | Contents |
| --- | --- |
| `input/` | Two genomic fragments, two supplied CDS sequences and two protein FASTA files; all six files are byte-identical to the supplied originals |
| `archived_annotation/` | All eight supplied annotation files, including two-query BLAST XML, InterProScan JSON, annotation tables and historical structure summaries |
| `paper_model_output/case2.md` | Original manuscript model report, byte-identical to the supplied file, in its original Chinese language |
| `paper_model_output/README.md` | Provenance and separate reading notes; the original report is not edited |
| `expected_replay/FLM_wt/`, `expected_replay/FLM_alt/` | Parsed annotation snapshots produced by v0.3.0 from the archived raw evidence |
| `replay.py` | One command to verify the archive and replay both proteins separately |
| `provenance/` | Original/public file hashes, sequence checks, code fingerprints and validation record |
| `recovered_server_archive/` | Original AF3 ZIP exports, ten converted models, selected structures, reference structure, historical reports and independent evidence checks |
| `../manifest.json`, `../checksums.sha256` | Published-file integrity records in the parent `examples/` directory |

In the original `archived_annotation/` snapshot, historical absolute server paths in four files were replaced with portable database labels or `UNAVAILABLE_STRUCTURE/` references. These original publication records remain unchanged and do not describe the availability of the newly added supplement. The native CIF files and corresponding PDB structures are now supplied in `recovered_server_archive/`. Every file from the original 15-file archive remains accounted for in `provenance/source_manifest.json`; provenance for the recovered material is recorded separately under `recovered_server_archive/audit/`.

## Read the manuscript result

Open [`paper_model_output/case2.md`](paper_model_output/case2.md). No database or API key is required to read the completed report. The original provider response envelope, model identifier, original prompts and generation settings were not included in the supplied archive, so these are recorded as unknown rather than reconstructed. This does not imply that the author's model analysis was not performed.

The original report analyzes the pair. Pipeline v0.3.0 is a single-protein workflow: the replay below processes the two inputs separately and does not regenerate the historical paired interpretation or invent four model-arm outputs.

## Verify and replay

Place the `examples/` directory inside the existing `pipeline/` directory. From `pipeline/`, use Python 3.10 or newer:

```bash
python examples/arabidopsis_flm/replay.py --verify-only

PPT_FLM_RUN="$HOME/ppt_flm_replay_$(date +%Y%m%d_%H%M%S)"
python examples/arabidopsis_flm/replay.py --out "$PPT_FLM_RUN"

cat "$PPT_FLM_RUN/replay_status.json"
cat "$PPT_FLM_RUN/FLM_wt/annotation/annotation_summary.tsv"
cat "$PPT_FLM_RUN/FLM_alt/annotation/annotation_summary.tsv"
```

The output directory must be new or empty and outside the example directory. A successful run ends with `PASS: both annotations match the bundled reference records; no database searches or model calls performed.` The replay status must be `completed`. Errors return a nonzero exit code and are not treated as completion.

If the pipeline source is elsewhere, supply `--pipeline-dir /absolute/path/to/pipeline`. The integrity-only check can be used without the pipeline source. There is no need to install Conda or databases just to run this standard-library replay when a suitable Python is already installed.

Expected results:

| Protein | Length (aa) | BLAST hits | Domain/signature records | GO terms | Structure imported by annotation replay |
| --- | ---: | ---: | ---: | ---: | --- |
| FLM_wt | 94 | 5 | 8 | 5 | not imported |
| FLM_alt | 85 | 5 | 8 | 5 | not imported |

The eight domain/signature records include overlapping annotations from different member databases; they do not imply eight independent physical domains. The replay checks full parsed JSON records, not only these counts. It also retains fresh input bindings, split XML provenance and evidence manifests in the run directory.

The XML contains two queries. The script selects the matching `Iteration` for each protein and adjusts root query header fields; it preserves all of that query's hits, alignments and database statistics. InterProScan records are selected by exact protein sequence. Source files are never overwritten. A successful replay verifies parsing and input-to-evidence correspondence; it does not rerun the historical searches or validate phenotype causality.

## Verify the recovered structures

From `pipeline/`, run the separate check:

```bash
python examples/arabidopsis_flm/recovered_server_archive/verify_recovered_evidence.py
```

This standard-library check verifies the recovered file hashes, both canonical CDS-to-peptide translations, the AF3 job input sequences, all ten native CIF-to-PDB conversions, and selected-model confidence summaries. It does not rerun AlphaFold, structural alignment, database searches, splice prediction, or language-model inference.

The original `replay.py` remains an annotation-only replay. Its structure field is still missing by design because it does not import the recovered models; this is distinct from whether structure files are available in the repository. See the [recovered-evidence guide](recovered_server_archive/README.md) for score definitions and historical interpretation notes.

## Recorded methods and versions

The archived XML records **BLASTP 2.16.0+** and database label **refseq_protein**. The archived InterProScan JSON records **5.67-99.0**. The exact historical database release is not supplied. The current v0.3.0 installer defaults to nr and InterProScan 5.78-109.0, which are different resources. Do not describe a new run with those defaults as an exact repetition of this historical annotation.

The raw BLAST XML explicitly records BLOSUM62, E-value 1e-5, gap opening 11, gap extension 1 and filter F. Other historical command-line options cannot be established solely from these files. v0.3.0 default values recorded in a replay request are parser invocation metadata, not proof of unrecorded historical search parameters.

The original 15-file archive contained historical structure summaries without their referenced coordinate files. The 2026-10-20 supplement now supplies the original AF3 CIF exports, converted PDB models, confidence records and the AFDB_Q38836 reference structure. The supplied structure check reproduces the selected-model mean confidence from coordinates. The historical TM-scores compare each protein with AFDB_Q38836, not WT with ALT; their original alignment outputs and normalization settings remain unavailable. The existing annotation replay and its reference outputs are unchanged.

The historical summary retains three BLAST hits per protein and zero GO scores. The v0.3.0 replay reads all five hits in the raw XML and recovers five GO terms from the raw InterProScan entries. These parsing differences are intentional and documented; historical files are not changed to match the new parser.

## Sequence and interpretation notes

The six input FASTA files are preserved exactly as supplied. Standard-code translation from the first nucleotide to the first stop yields the supplied 94-aa and 85-aa proteins respectively. The CDS files differ by a 17-nt insertion, `TTTTTGGTGTTATGAAG`, after nucleotide 243 of the WT file. The provided WT CDS has 637 nt and a first stop at codon 95; the ALT CDS has 654 nt and a first stop at codon 86. These observations are recorded without modifying the biological inputs or inferring full-length canonical isoform identities from filenames.

The two proteins share the first 81 residues. The remaining WT sequence is `DRSAVQAGETSMF`; the remaining ALT sequence is `FLVL`. Their net length difference is nine residues, but the original report's statement that the first 85 residues are identical followed by a simple nine-residue deletion does not match the supplied FASTA files. This correction is recorded separately; the original report is preserved verbatim.

## Literature provenance

The author identifies the input CDS sequences with the following study:

Hanemian M., Vasseur F., Marchadier E., et al. **Natural variation at FLM splicing has pleiotropic effects modulating ecological strategies in Arabidopsis thaliana.** *Nature Communications* 11, 4140 (2020). [https://doi.org/10.1038/s41467-020-17896-w](https://doi.org/10.1038/s41467-020-17896-w).

The paper describes an intronic substitution in Can-0 that promotes inclusion of 17 nucleotides and premature termination. This establishes relevant experimental context, not independent verification of each packaged CDS base or every hypothesis in the archived model report. The author-supplied source attribution is retained; exact CDS-to-supplementary-record matching remains undocumented in this package. The genomic sequences match the author's earlier pasted sequences labeled Supplementary Data 1, sheet Feuil1, cells A3 and A5; the original supplementary workbook was not independently compared during packaging.

The paper's stated license is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the paper when using its data and retain the relevant attribution. This note does not assign a new license to the author's report or override the repository's licensing terms. Literature background and the historical report are kept separate from raw annotation input.
