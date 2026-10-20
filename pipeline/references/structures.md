# Optional structure processing

These commands operate on existing predicted/experimental models. They do not run AlphaFold 2/3 or create a structure when none exists. The core annotation/trait pipeline can run with structure explicitly missing. Activate the supplied environment (Biopython 1.85 is used for conversion and chain checks).

## 1. Retrieve an existing AlphaFold DB model

Use a real UniProt accession obtained for your protein:

```bash
python 1_afdb_seq_search_download.py --fasta "$PPT_FASTA" \
  --accession YOUR_UNIPROT_ACCESSION --outdir /path/to/new/afdb_output
```

Alternatively select references by a Swiss-Prot search:

```bash
python 1_afdb_seq_search_download.py --fasta "$PPT_FASTA" \
  --local-db /path/to/swissprot --top-k 5 --outdir /path/to/new/afdb_references
```

A supplied `--blast-xml` or explicit `--online-blast --email YOUR_EMAIL` is also supported. The main nr installation is not automatically treated as a Swiss-Prot database; use an accession to avoid a separate reference search. Read `models.json` for retrieved files and their roles. A homolog model is not a target structure. No API key is required by this AFDB retrieval path. Remote access is required even if the reference BLAST search was local. There is no hard-coded AFDB model-version fallback.

## 2. Convert existing mmCIF predictions

```bash
python 2_af3_cif_to_pdb.py --input /path/to/existing/cif_or_zip_directory \
  --output /path/to/new/converted_pdb
```

Input may be one CIF/mmCIF, `.cif.gz`/`.mmcif.gz`, ZIP or a directory. Output must be outside the input tree. Converted filenames use source-derived identifiers to avoid collisions. Source hashes and filenames are in `conversion.json`. Unsafe ZIP paths and oversized input are rejected. PDB cannot represent every mmCIF feature; multi-character chain identifiers are rejected explicitly rather than silently renamed. Retain the original mmCIF files.

## 3. Match the exact target chain

```bash
python 3_map_af3_pdbs.py --fasta "$PPT_FASTA" \
  --pdb-dir /path/to/converted_pdb --output /path/to/new/mapped_target
```

The first model is read and each complete chain is compared with the full protein sequence, including isolated residues. Only exact matches are exported, one chain per file. `mapping.json` records the selected chain, source and hashes. A missing residue, truncated model or homolog-only match does not become a target match. Inspect the mapping before choosing a structure; multiple valid models are not arbitrarily ranked.

## Attach one verified target PDB

Add these options to `4_cotton.py` or `llm.py pipeline`:

```text
--structure-pdb /path/to/mapped_target/selected_target.pdb --structure-origin alphafold3
```

Declare the actual origin: `alphafold2`, `alphafold3` or `experimental`. If more than one chain matches, supply `--structure-chain`. The collector rechecks the entire chain; the user's origin declaration is retained as such. Predicted-model pLDDT is summarized from one CA per residue; experimental B-factors are never treated as pLDDT. Experimental coordinates are archived but do not add a predicted-confidence item to model evidence. No TM-align, Foldseek, structure-similarity ranking or phenotype score is computed.

AFDB retrieval can instead be requested directly with `4_cotton.py --afdb-accession`, but the strict local workflow does not silently call that remote service. Download once and attach the resulting exact target PDB when a local annotation run is required.

Format support follows [Biopython MMCIFParser](https://biopython.org/docs/1.85/api/Bio.PDB.MMCIFParser.html) and [PDBIO](https://biopython.org/docs/1.85/api/Bio.PDB.PDBIO.html). AFDB retrieval uses its [public API](https://alphafold.ebi.ac.uk/api-docs).
