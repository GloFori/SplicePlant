# Reproducible examples

[Arabidopsis FLM](arabidopsis_flm/README.md) contains the supplied case-study inputs, archived annotations, the author's original manuscript model report, and an executable annotation replay for pipeline v0.3.0.

The manuscript model analysis has already been performed, as confirmed by the author. Its supplied report is preserved unchanged in [`paper_model_output/case2.md`](arabidopsis_flm/paper_model_output/case2.md). Reading and checking this example does not require another model call or an API key.

From the `pipeline/` directory:

```bash
python examples/arabidopsis_flm/replay.py --verify-only
python examples/arabidopsis_flm/replay.py --out "$HOME/ppt_flm_replay_$(date +%Y%m%d_%H%M%S)"
```

The replay uses archived BLAST and InterProScan evidence, checks both protein sequences, and compares the complete parsed annotation records with the included reference outputs. It requires Python 3.10+ and the adjacent v0.3.0 pipeline source. It does not require installed annotation databases, BLAST/InterProScan executables, Internet access, or a model API key.

This archived case and the v0.3.0 replay have separate provenance. The original report is not relabeled as a new four-arm v0.3.0 model run. See the case README for observed database versions, the available evidence, and interpretation notes.

## Recovered FLM structures

The original WT and ALT AlphaFold Server exports and associated historical evidence are now included in [`arabidopsis_flm/recovered_server_archive/`](arabidopsis_flm/recovered_server_archive/README.md). The existing annotation-only replay remains unchanged. To verify the added structure evidence, run from `pipeline/`:

```bash
python examples/arabidopsis_flm/recovered_server_archive/verify_recovered_evidence.py
```

This check requires Python 3.10+ and the standard library only. It checks archived files without new predictions, database searches or model calls.
