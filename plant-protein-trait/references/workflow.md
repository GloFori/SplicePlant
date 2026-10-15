# Execution workflow

Run commands from the source/skill directory, or use absolute paths to its scripts. All output directories in this guide are examples; choose fresh directories for retries. The caller must supply an actual protein FASTA and an actual target trait.

## 1. QC

```bash
python scripts/check_sequence.py --fasta local_data/protein.fa --output runs/qc.json
```

Optional `--expected-id` and `--expected-length` enforce a known identity/length. Only the 20 canonical amino acids are accepted; stops and ambiguous letters cause failure. Input is uppercased, not trimmed or repaired. Exactly one record is required. This is protein QC, not a genome/CDS translation audit.

## 2. Choose evidence sources

Default behavior is offline. Omitting a stage leaves it explicitly `not_run`; an empty successful BLAST search is `completed_no_hits`. Collection errors return a nonzero exit code, preserve partial raw output and do not create a completed evidence manifest.

### Existing results (recommended for archived reproduction)

```bash
python scripts/4_cotton.py --fasta local_data/protein.fa --output runs/annotation \
  --blast-xml local_data/blast.xml --interpro-json local_data/interproscan.json
```

BLAST XML/XML2 must contain one query. Length and all aligned query residues are checked. InterProScan JSON must contain exactly one matching full sequence. Parsing validates compatibility, not the authenticity of externally supplied measurements.

### Existing local databases

```bash
python scripts/4_cotton.py --fasta local_data/protein.fa --output runs/annotation \
  --blast-db /path/to/existing/swissprot --pfam-db /path/to/existing/Pfam-A.hmm \
  --interpro-json local_data/interproscan.json --threads 2
```

`blastp` and `hmmscan` must be in PATH, or supply `--blastp`/`--hmmscan`. Pfam must already be prepared for HMMER. Archive database version/release and input database checksums separately; an absolute database path alone is not a reproducible version. This package never downloads or builds these databases. HMMER keeps domains with independent domain E-value ≤0.001; overlapping InterPro/Pfam signatures remain annotated observations, not independent support counts.

### Online annotation (explicit sequence submission)

```bash
python scripts/4_cotton.py --fasta local_data/protein.fa --output runs/annotation \
  --online-blast --remote-database swissprot --interpro-remote \
  --email "$ANALYSIS_CONTACT_EMAIL" --max-wait 1800
```

Use the researcher's own contact email. Do not copy an address out of old code. Both services receive the protein sequence; no study trait is sent. NCBI defaults to Swiss-Prot here, not nr. A remote database override is explicit. NCBI polling is at least 60 seconds per job. Each HTTP request is bounded and the polling window is finite; final network calls can extend wall time slightly beyond the polling window.

If a job is pending, job IDs and responses are retained in `raw/`. Download the completed BLAST XML/InterProScan JSON from the relevant service and import it into a **new** output directory; do not repeatedly resubmit the same sequence or fabricate an empty result. Automatic remote-job resumption is not implemented in 0.1.0.

### Optional target structure

Add `--afdb-accession ACCESSION` to annotation when a UniProt accession is known. The API's full sequence and bounds must exactly match the target before a model is assigned as query evidence. A different sequence is recorded as `reference_only`; target pLDDT remains null. The API-provided download URL is used, without hard-coded model versions. Mean pLDDT uses one CA per residue from the first model; this is only valid for a predicted model that encodes pLDDT in B-factor fields.

No de novo AlphaFold prediction, Foldseek search or TM-align execution is automated in this release. An unavailable model may be omitted and recorded as missing. This avoids calling a homolog's predicted model the target structure.

## 3. Freeze protein evidence and prepare all four prompts

```bash
python scripts/new_llm.py prepare --fasta local_data/protein.fa \
  --annotation-dir runs/annotation --out runs/inference \
  --trait "your actual target trait" \
  --forbidden "study-specific label" --forbidden "another excluded background phrase"
```

The paired protocol requires the target trait. Only targeted prompts receive it; blind prompts and protein evidence are checked against it and the excluded phrases. Default excluded study terms include GWAS, SNP, sQTL, genotype and PSI. This is a literal case-insensitive leakage check, not a semantic proof of blindness. Include trait synonyms, cohort labels and study identifiers as extra exclusions. If legitimate database biology triggers the check, document that strict blinding cannot be maintained from that input rather than silently redacting evidence or claiming an independent prediction.

The protein label is neutralized as `query_001`. Taxonomy in homolog records remains valid protein evidence. The four prompts include the same normalized protein packet, never each other's results. Long evidence raises an error rather than being silently truncated. `query_mapping.json` and `preparation.json` are audit files; they must not be appended to blind requests.

Legacy `annotation_full.json` from the original `4_cotton.py` may be read directly. It may lack sequence hashes; that limitation is retained as `binding_warning`. Do not invent a hash for the old annotation and call it verified. The new collector binds results to the target sequence and creates hashes.

## 4. Execute a configured backend

Ollama:

```bash
ollama list
python scripts/new_llm.py run --prepared runs/inference --backend ollama \
  --model "$MODEL_ID" --num-ctx 32768 --max-tokens 6000
```

The service must already be running and the model installed. This package does not download weights or launch GPU jobs. Default endpoint is localhost:11434. A remote service should use HTTPS or a localhost SSH tunnel. A conservative byte-based input reserve guards against small Ollama context settings; check the model's actual capacity as well.

DashScope:

```bash
# Configure DASHSCOPE_API_KEY outside version control using your normal secret-management method.
python scripts/new_llm.py run --prepared runs/inference --backend dashscope --model "$MODEL_ID"
```

Default endpoint is `https://dashscope.aliyuncs.com/compatible-mode/v1`; override with `DASHSCOPE_BASE_URL` or `--base-url` if your account uses a different endpoint. The model must support the chat-completions parameters used by this client. No hosted model ID is bundled or assumed available.

Compatible endpoint:

```bash
# Configure OPENAI_API_KEY and OPENAI_BASE_URL outside version control.
python scripts/new_llm.py run --prepared runs/inference --backend compatible --model "$MODEL_ID"
```

Each call has exactly a system message and a user message. Blind verify/explore are executed first, then targeted verify/explore. Temperatures are recorded as 0.2/0.5. `--analysis blind|targeted` and `--mode verify|explore` can select unattempted slots. Successful or failed slots are never overwritten; create a fresh preparation directory for retries and retain prior attempts. There is no silent output repair or provider fallback. Empty, token-limited, invalid-JSON, invented-evidence-ID or unsupported direct-evidence results fail validation. Raw successful HTTP responses remain available for review.

## 5. Freeze and compare

```bash
python scripts/new_llm.py freeze --prepared runs/inference
python scripts/summarize.py --prepared runs/inference --output runs/comparison
```

Only after all four successful results are frozen may independent study background be introduced:

```bash
python scripts/summarize.py --prepared runs/inference --output runs/comparison_with_context \
  --background local_data/independent_study_context.txt
```

This copies the background separately; it does not rerun or rewrite model results. The final biological interpretation is a scientific review, guided by [interpretation.md](interpretation.md), not an automatic causal claim from the summarizer.

## 6. Archive for review

Archive `annotation/`, `inference/`, `comparison/`, exact commands, database versions, code commit/tag and your environment report. Preserve raw unsuccessful attempts. Put credentials nowhere in the archive. Replaying frozen evidence isolates the model analysis from database drift; comparing a fresh live search is a separate analysis.

API references: [NCBI URL API](https://blast.ncbi.nlm.nih.gov/doc/blast-help/urlapi.html), [NCBI usage rules](https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html), [EBI InterPro sequence search](https://www.ebi.ac.uk/training/online/courses/interpro-functional-and-structural-analysis/how-to-search-interpro/sequence-searching/), [AFDB API](https://alphafold.ebi.ac.uk/api-docs).
