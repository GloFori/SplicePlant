# Real example to be supplied by the maintainer

No real Arabidopsis example has been selected or analyzed in this release. No sequence, accession, target trait or model result is fabricated to fill that gap. The smoke test creates clearly labeled synthetic data at runtime; it is not an Arabidopsis example.

When adding the intended example, create `examples/arabidopsis/` and include:

1. One protein FASTA with the exact transcript/protein identifier and expected length.
2. AGI/UniProt accession, assembly/annotation release, source URL, retrieval date, and file/sequence SHA256 values. If a CDS/GFF/transcript is included, state the coordinate conventions and whether translation was verified.
3. Raw BLAST and InterProScan outputs, query-binding checks, database versions and any target structure provenance.
4. The target trait for the targeted arm and predeclared blind-exclusion terms. Keep these outside the protein evidence files.
5. Exact commands and source code commit/tag.
6. Four untouched model outputs and prompt files, actual model IDs, generation settings, timestamps and manifests.
7. A separately written A/B interpretation, missing-data statement and limitations.
8. Any independent study background stored outside upstream annotation and introduced only after the four outputs were frozen.

Do not put secrets or server-only paths in the example. Do not copy old cotton results into the Arabidopsis directory. A case whose model calls were not performed must say `prepared_not_executed`, not “reproduced.”

Publish a real-example release only after the complete commands have been tested on a clean machine/server and the scientific interpretation has been reviewed. Preserve the original software-only release for provenance.
