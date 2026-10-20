# What stays the same between routes

Both routes use the original annotation-method lineage: **BLAST + InterProScan → common evidence parsing → upgraded four-arm protocol → freeze → comparison**. They differ in where annotation is computed. The original seven-file hashes and earlier corrective refactoring remain documented in [the audit](../docs/AUDIT.md). This release does not restore proxy structure scores, silent failed-stage fallbacks or sequence alteration from the legacy scripts.

Shared explicit BLAST settings are nr, five hits, E-value 1e-5, BLOSUM62, word size 3, gap costs 11/1, composition-based statistics 2 and SEG off. These follow the original nr/five-hit/E-value defaults, with formerly implicit options made explicit on both routes. Overrides apply through the same argument names. Raw XML records the observed database/program information.

Both routes request InterProScan JSON, GO and pathways. By default both request all available applications as the original scripts did. To fix a narrower selection, pass the same `--interpro-applications Pfam,SMART,...` list on both routes. The installed/service versions may expose different application sets or releases; matching requested selection alone is not proof of identical evidence. Observed member database versions are saved; absent release information stays unknown.

Independent HMMER/Pfam was optional in the original code. It remains optional via `--pfam-db` locally. To include that supplement in an online-vs-local comparison, import the same real `--pfam-domtbl` file into the online route, or omit this extra stage on both routes. A Pfam-only scan is not silently substituted for full InterProScan. Supplemented online runs must be described as using imported supplementary evidence.

## Measure equality

```bash
python llm.py compare \
  --left /absolute/path/local_case/annotation \
  --right /absolute/path/online_case/annotation \
  --output /absolute/path/route_comparison.json
```

The command verifies each evidence manifest and the protein identity, compares normalized evidence and requested settings, and reports observed version differences. `matched` means matching actual evidence and requested analysis settings with completed core stages; it does not assert that unknown database versions are equal. Nonmatching or incomplete evidence returns a nonzero exit code and a comparison file. Do not discard evidence to make a comparison pass.

The same evidence and trait produce byte-identical prepared messages through the shared normalizer; source paths, arrival order and service job IDs do not enter the model input. Different evidence may produce different conclusions. Independent hosted-model generations can vary even with identical messages. Exactly reproducing the original reported text means retaining its raw model response and verifying its hashes, not asking a model to regenerate identical wording.

## Archive a manuscript case

Include the protein FASTA, code commit/release, exact commands, `workflow_request.json`, `annotation/`, `inference/`, `comparison/`, local `resources.json`/environment export when applicable and any recovered job files. Do not include API keys. Keep failed/uncertain attempts separate. The example should provide actual archived raw BLAST XML and InterProScan JSON so that reviewers can first run the replay route without waiting for public annotation services. No biological demo or model results are fabricated in this source release.
