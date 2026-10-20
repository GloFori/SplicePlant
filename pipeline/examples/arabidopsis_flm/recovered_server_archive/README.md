# FLM: recovered structures and historical interpretation evidence

This additive package uses the existing SplicePlant FLM inputs at commit `94ae36cdeb7bad110da96e90afcec355943400bb` as the canonical sequence set. It supplements the published example with recovered historical structure-prediction exports and reports. No new predictor, database search, AlphaFold job or language-model call was run.

## Canonical inputs

`canonical_inputs/` contains byte-identical copies of the six published genomic, CDS and protein FASTA files. WT and ALT CDS lengths are 637 and 654 nt; translation to the first stop gives the supplied 94-aa and 85-aa proteins. The recovered server ALT-labelled CDS incorrectly duplicates WT and is excluded from this package. Keep the repository's existing `input/` files unchanged. `audit/canonical_input_sources.json` records the frozen GitHub URLs, Git blob IDs and SHA-256 hashes.

The two proteins share their first 81 residues, followed by WT `DRSAVQAGETSMF` and ALT `FLVL`. The observation is a changed C-terminal sequence with a net shortening of nine residues; it is not an identical first 85 residues followed by a simple nine-residue deletion.

## What the recovered material adds

| Material | Evidence supplied |
| --- | --- |
| `raw_af3/fold_flm_wt.zip`, `raw_af3/fold_flm_alt.zip` | Unchanged AlphaFold Server exports: five native CIF models per protein, original confidence JSON, structure job requests and seeds, MSAs, templates and output terms |
| `converted_pdb/` and `mapped_structures/` | Supplied PDB conversions and portable copies of the originally selected model-0 links; all ten structures match the canonical protein sequences and the native CIF atoms |
| `reference_structures/AFDB_Q38836.pdb` | The reference structure named as the comparison partner in both historical TM-score records |
| `historical_annotation/` | Portable copies of the historical structure summaries and structured protein-evidence export, with numeric values retained |
| `historical_model_outputs/` | Three unchanged Chinese report/evidence files recovered from `out/`; these are archived model outputs, not newly validated biological findings |
| `audit/` | Input provenance, ten-model checks, per-residue confidence, score definitions and original file hashes |

`final_reports_case2.json` is a structured protein-annotation export. It is not the original language-model API response envelope. Historical report files are separate from the previously published manuscript report in `../paper_model_output/`.

## Reproduce the evidence checks

From this directory run:

```bash
python verify_recovered_evidence.py
```

The check requires Python 3.10 or newer and the standard library only. It verifies checksums, canonical CDS-to-protein translations, AF3 request-to-protein bindings, ten native-CIF-to-PDB conversions, selected-model confidence summaries and the definition of the recorded TM-score partner. It does not rerun AlphaFold, the historical structural alignment, the splice predictor or a language model. Keep the existing annotation-only replay and its historical status unchanged.

## Score definitions and interpretation boundaries

- The selected WT/ALT historical mean atom-level confidence values, 87.7790 and 89.0102, are reproduced from PDB/CIF B columns. A mean using one CA per residue instead gives 92.2661 and 93.3374. These are distinct aggregation methods.
- Historical composite scores 93.8895 and 94.5051 follow `historical_annotation/pipeline_summary.txt`; they are not pLDDT. Native full-data JSON atom confidence values differ slightly from CIF B-column values (maximum absolute difference at most 0.1 across the recovered models); both representations remain labelled and unchanged.
- Historical TM-scores 0.82494 (WT) and 0.8594 (ALT) refer to comparison with AFDB_Q38836. They are not a WT-versus-ALT TM-score. The original alignment output and normalization settings were not recovered, so these are archived reported values rather than independently recomputed alignments.
- Protein interaction changes, dominant-negative action, protein stability, transcript decay and phenotypic effects are not established by predicted structure confidence. Preserve raw historical interpretations, including their limitations, and use evidence-qualified statements in the manuscript.

## What is now available and what remains absent

The native structures, original structural confidence files and structure job inputs are now available and can be bound to the canonical proteins. This resolves their absence from the earlier public example. Original language-model prompts, provider/model identity, generation parameters and API response envelope were not found. The AF3 job requests are structure-prediction requests, not language-model prompts. No splice-site probability output or original structural alignment output was recovered.

When integrating this directory, update the parent README's statement that native structures are unavailable and rebuild the parent integrity manifests. Do not describe the historical annotation-only replay as structure-enabled unless it is explicitly changed and validated. Preserve the terms supplied inside the original AlphaFold ZIP files. This package assigns no new licence to the outputs.
