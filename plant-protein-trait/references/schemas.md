# Input, evidence and result schemas (v1.0)

## Annotation interface

`annotation_full.json` is the required rich input. Preferred shape:

```json
{
  "PROTEIN_ID": {
    "seq_id": "PROTEIN_ID",
    "length": 80,
    "sequence_sha256": "SHA256_OF_ACTUAL_UPPERCASE_SEQUENCE",
    "blast_hits": [],
    "domains": [],
    "go_terms": [],
    "structure": {"status": "missing", "avg_plddt": null, "tm_score": null},
    "status": {"blast": "not_run", "domains": "not_run", "go": "not_run"}
  }
}
```

The example above is a schema illustration, not a protein result. Do not create fabricated evidence to populate it.

The loader also accepts a record list, `{ "records": [...] }`, or one object with `seq_id`. It selects the exact FASTA ID, not the first record or highest score. Length and a present sequence hash must agree. `interpro_domains` is accepted as a legacy alias for `domains`; structure aliases `avg_pLDDT`, `plddt` and `best_tm_score` are normalized. `structure_info.json` can supplement the selected record. Original `reason=proxy_score` data are excluded from structural evidence.

`annotation_summary.tsv` alone is insufficient: the original file only contains heuristic scores and cannot substantiate a function. No `annotation_ranked*` or `final_report*` is required. Old command-line flags are not all maintained; use the migration table in `docs/AUDIT.md`.

## Raw evidence semantics

- BLAST: XML and XML2, one query; source version and database retained. `identity_pct`, `query_coverage_pct`, `e_value`, `bit_score` refer to the best bit-score HSP. `query_coverage_union_pct` merges all HSP query positions. Raw score is a separate value. Aligned query residues must match the FASTA, including gap removal only from the alignment text.
- InterProScan: JSON `results[].sequence` must equal the query. Signatures retain InterPro ID, source database/release, score, E-value when supplied and locations. GO terms are found in nested `signature.entry.goXRefs` as well as applicable result/match fields; they are computational transfers.
- Pfam: local hmmscan domtblout; retain domain coordinates and independent domain E-values. There is no automatic GO mapping from a bare Pfam name.
- Structures: `missing`, `reference_only`, `legacy_unverified` or `available`. Legacy numeric fields without an explicit verified status remain `legacy_unverified` and are excluded from model observations. The new AFDB path verifies full-length sequence identity and the downloaded model's residue count. Missing numeric fields are `null`, not invented zeros. When importing third-party annotations marked `available`, independently review their sequence binding and prediction provenance.

## Prepared evidence packet

`protein_evidence.json` contains one neutral-ID query and ordered observations. IDs `E001`, `E002`, ... are assigned first to homology, then domain, then GO, then available target-structure evidence. Each observation retains its type and explicit `experimental_on_query=false`. Trace each observation back to the corresponding rich annotation list; retain the source annotation file and its manifest.

Study context, heuristic scores, prior LLM descriptions and file paths are not passed to inference. The packet and messages are frozen with SHA256. `query_mapping.json` retains the original ID for auditing outside the model prompt.

## Model result contract

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `analysis`, `mode` | `targeted`/`blind`, `verify`/`explore`; must match requested slot |
| `molecular_function`, `pathway` | Evidence-constrained interpretation, or unknown |
| `verdict` | Targeted: supported / weak_support / unsupported / insufficient; blind: not_applicable |
| `top5` | Exactly five ranked rows, using unknown/insufficient if fewer defensible hypotheses exist |
| `evidence_assessment` | Existing evidence ID, claim supported, grade and limitation |
| `strongest_link`, `weakest_link` | Concise mechanistic assessment |
| `contrary_evidence`, `limitations`, `proposed_tests` | Lists |

Each Top-5 row has `rank`, `biological_process`, `potential_trait`, `supporting_evidence`, `confidence`, `evidence_grade`, and a concise evidence-linked `reasoning_chain`. Confidence is high / moderate / low / insufficient. Grades are direct_evidence / indirect_mechanistic_evidence / biological_plausibility / speculative_connection. No evidence means insufficient confidence. Invented evidence IDs fail validation. This release's protein-annotation inputs do not supply direct experimental phenotype evidence, so upgrading them to direct evidence fails validation.

Validation is structural, not a guarantee of biological correctness. Raw model responses remain unchanged even if validation fails. The package does not silently rewrite results or try another prompt until it obtains a desired conclusion.

## Status and manifests

- `prepared_not_executed`: prompts exist; there is no model result.
- `started`: an attempt began; an interruption can leave this state.
- `failed`: request, parsing or validation failed; no successful result is claimed.
- `completed`: one valid provider result was recorded; scientific review is still required.
- `four_results_frozen`: all four slots completed and their files were hashed.

Manifests record UTC timestamps, Python version and file digests. They detect accidental changes relative to the saved manifest, not malicious rewriting of both file and manifest. Preserve immutable repository releases or institutional archives for publication.
