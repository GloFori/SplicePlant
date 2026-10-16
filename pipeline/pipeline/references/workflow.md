# Shared analysis workflow

The maintained entry points are `llm.py pipeline` for one-command execution, `4_cotton.py` for annotation and `llm.py` for the four independent inference arms. Read [local](local.md) or [online](online.md) instructions for choosing the annotation route.

## Configure a model

Set the actual model identifier available in your account. For DashScope, enter the key in the terminal, not in chat or a committed file:

```bash
MODEL_ID='replace-with-your-actual-model-id'
read -r -s -p "DashScope API key: " DASHSCOPE_API_KEY
printf '\n'
export DASHSCOPE_API_KEY
```

Run the `read` line and then type the key at its prompt. Text after `-p` is only a prompt, not a variable assignment. The default endpoint is `https://dashscope.aliyuncs.com/compatible-mode/v1`; use `DASHSCOPE_BASE_URL` or `--base-url` for an account with another endpoint. The model must accept the client's Chat Completions fields: nonstreaming messages, temperature and max_tokens. No model ID or model availability is assumed.

A compatible provider uses `--backend compatible`, `OPENAI_API_KEY` and `OPENAI_BASE_URL`. Ollama uses `--backend ollama --model YOUR_INSTALLED_MODEL` and an existing running service, default localhost:11434. Model installation and GPU scheduling are not silently launched. Model context limits are checked conservatively; see CLI options.

## One-command execution

Both `--mode local` and `--mode online` perform:

1. Protein QC and BLAST/InterProScan evidence collection.
2. Evidence freezing and four isolated prompt preparations.
3. Blind verify/explore, then targeted verify/explore, with no shared chat history.
4. Validation and freezing of all four outputs, then comparison/report generation.

Append `--prepare-only` to stop before any model calls. Later repeat the same command with `--resume`, an actual `--model`, and without `--prepare-only`. Model settings can be configured until the first model attempt. Completed calls are verified and skipped; failed/uncertain calls are retained and never silently charged again. A deliberate model retry uses a new inference preparation directory and the same archived annotation.

## Replay archived annotation

```bash
python llm.py pipeline --mode replay \
  --fasta /absolute/path/protein.fa --trait "your target trait" \
  --blast-xml /absolute/path/blast.xml \
  --interpro-json /absolute/path/interpro.json \
  --backend dashscope --model "$MODEL_ID" \
  --out /absolute/path/replay_run
```

Match `--database`, `--max-hits`, `--seg` and other settings to those used for the archived search. Declared settings do not retrospectively prove how imported raw data were generated. Sequence length/aligned residues and the InterPro full sequence are validated. Preserve original job/version records with imported evidence.

## Individual stages and old recovered jobs

```bash
python llm.py qc --fasta /absolute/path/protein.fa
python llm.py prepare \
  --fasta /absolute/path/protein.fa \
  --annotation-dir /absolute/path/annotation \
  --out /absolute/path/inference --trait "your target trait"
python llm.py run --prepared /absolute/path/inference \
  --backend dashscope --model "$MODEL_ID"
python llm.py freeze --prepared /absolute/path/inference
python llm.py summarize --prepared /absolute/path/inference \
  --output /absolute/path/comparison
```

Exactly one canonical-amino-acid protein is supported. Stops, ambiguous amino acids and DNA-only alphabets are rejected rather than silently converted. Supply actual CDS/isoform reconstruction separately if starting from a genomic sequence.

The target trait is required for the paired protocol. Add study-specific exclusions using repeated `--forbidden` flags. Blind requests do not contain the target trait or study background. If legitimate annotations contain the target phrase, the strict blindness check stops; do not remove factual evidence to manufacture a blind prediction.

Missing annotation remains missing; an empty successful search is different from a network failure. No aggregate proxy confidence/structure score is generated. Only after all four results are frozen may study background be copied into a post-freeze report with `llm.py summarize --background FILE`.

## Preserve results

Keep code commit/release, software/database versions, `workflow_request.json`, annotation/raw results, prompt files, provider metadata/raw responses, frozen manifests and comparison. Review scientific claims independently; valid JSON does not establish biological correctness. See [route equality and archiving](reproducibility.md).
