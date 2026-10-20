# Online annotation without local databases

Python 3.10+ is sufficient. No BLAST, HMMER, InterProScan or local annotation database is needed. Both services receive the protein sequence and your contact email. Model credentials are separate; see [shared workflow](workflow.md).

```bash
ANALYSIS_CONTACT_EMAIL='replace-with-your-own-email'
python llm.py pipeline --mode online \
  --fasta /absolute/path/protein.fa --trait "your target trait" \
  --database nr --email "$ANALYSIS_CONTACT_EMAIL" \
  --backend dashscope --model "$MODEL_ID" \
  --out "$HOME/ppt_runs/online_case" --max-wait 1800
```

BLAST and InterProScan are submitted independently and polled in parallel. The original search default is nr, five hits, E-value 1e-5. Use the same explicit settings as the local route. `--max-wait` limits each service's polling window; bounded HTTP calls can extend elapsed wall time. WAITING/PENDING/RUNNING means the service has not returned results, not that a new task should be submitted.

## Resume instead of resubmitting

Repeat the SAME command with `--resume` appended. You may increase `--max-wait` and `--request-timeout`; do not change sequence or analysis settings. Saved task IDs and verified completed results are reused. Known transient polling failures are retried within the time limit. Job state files under `annotation/raw/` distinguish pending, connection errors, completion and failure.

If a submission request fails before an ID is received, its outcome is uncertain. The code stops rather than automatically creating another task. Recover its ID through the service, then supply `--blast-rid` or `--interpro-job-id` along with `--resume`. If no task can be recovered, an explicitly chosen new run directory starts a new attempt. Preserve the uncertain attempt.

Two processes must not write the same run at once. A `.lock` file prevents this. After a force-killed process, confirm it has stopped before removing its stale lock.

## Recover a v0.1 task

Old outputs have no v0.2 `request.json`. Use a new annotation output directory and the existing task ID. Match the settings actually submitted by v0.1 (Swiss-Prot, 20 hits, SEG on), rather than the new original-method defaults:

```bash
python 4_cotton.py --fasta /absolute/path/protein.fa \
  --output /absolute/path/recovered_annotation \
  --blast-rid "$EXISTING_BLAST_RID" \
  --database swissprot --max-hits 20 --seg yes \
  --interpro-remote --email "$ANALYSIS_CONTACT_EMAIL"
```

If InterProScan was already submitted separately, also supply `--interpro-job-id "$EXISTING_IPR_JOB"`. If its JSON was downloaded, replace `--interpro-remote` with `--interpro-json /absolute/path/interpro.json`. Retrieval is validated against the protein, but old search parameters cannot be independently proven from a user-supplied ID; that limitation is recorded.

After recovery, follow [the step-by-step workflow](workflow.md) to prepare and run inference. The new default nr search must not be falsely described as the old Swiss-Prot search.

Sources: [NCBI URL API](https://blast.ncbi.nlm.nih.gov/doc/blast-help/urlapi.html), [NCBI polling limits](https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html), [EBI Job Dispatcher](https://www.ebi.ac.uk/jdispatcher/).
