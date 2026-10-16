---
name: plant-protein-trait
description: 'Operate the SplicePlant single-protein function-to-trait pipeline on a Linux server: set up local software and databases, collect protein evidence, and run isolated targeted and blind model analyses. Use for reproducible reviewer runs; requires terminal access to the server and the accompanying pipeline checkout.'
---

# Plant protein to trait

This Skill operates the same seven-script workflow used by human reviewers. It does not contain a second analysis engine. A model with no server terminal/file access can explain the workflow but cannot execute it merely by reading this file.

## Locate the pipeline

Resolve this Skill's own directory first. Run `python <skill-dir>/scripts/ppt.py locate`. In the repository, the engine is the parent `pipeline/` directory. Installed copies use the binding created by `ppt.py install`; alternatively pass `--pipeline-root` before the command or set `PPT_PIPELINE_ROOT`. If the checkout moved, obtain its actual path rather than guessing or downloading another code version.

Read the located pipeline's `README.md` for the human workflow; use its `references/manual.md` for detailed commands. Both interfaces call those same implementations. The directory is named `pipeline`, matching the repository; do not create a separate `pipline` tree.

## Operate a local run

1. Establish one verified protein FASTA, target trait, installation/resource directory, result directory and actual provider/model. The current engine does not reconstruct a protein from a gene ID/genomic fragment and does not compare multiple isoforms. Run `llm.py qc --fasta ...` before analysis. Do not remove stops or concatenate inputs to pass QC.
2. Check Linux architecture and the intended storage location. The local distribution supports x86_64, requires Conda, and downloads large databases on first use. Respect the user's chosen compute node/scheduler. Never submit a long analysis to an unspecified cluster queue.
3. If the user has requested installation, use `ppt.py setup --root INSTALL_ROOT --with-databases --execute`. It creates a dedicated environment and invokes the database script. Without `--execute` it only prints the plan. Existing completed resources are reused; repeated installation requests do not authorize overwriting unrelated environments. Do not ask again for installation/download approval already given by the user.
4. Use the generated `INSTALL_ROOT/local_config.json`. Obtain API credentials through the server environment, never ask for their values in chat. Use `ppt.py run --mode local --config ... --fasta ... --trait ... --backend ... --model ... --out ...`. This checks evidence, prepares and executes four isolated calls, freezes successful results and writes a comparison. With unavailable credentials/model selection, use `--prepare-only` and clearly report that inference has not run. Resume uses the same input/settings and `--resume`.
5. Optional structure processing follows `references/structures.md` in the pipeline. Scripts 1–3 retrieve or process existing models; they do not run AlphaFold prediction. Do not run them as mandatory stages. Only exact full-chain target matches may be attached with `--structure-pdb` and the actual declared origin. Homolog models are references, not query structures. Missing structure stays missing; no invented TM-score or proxy score.
6. Online annotation is available when explicitly chosen; follow the pipeline's `references/online.md`. Never silently switch a failed local run to remote services or change nr to Swiss-Prot. Preserve partial output and task IDs; use verified resume rather than resubmitting uncertain jobs.

## Evidence and independent analysis

Targeted verify/explore and blind verify/explore are four separate stateless provider calls from the same frozen evidence. A conversation already given the trait is not a blind evaluator. Never substitute your own chat answer for a failed independent call. Study/locus/association background is added only after the four results are frozen.

Homology, domains, GO and model confidence are computational observations, not experimental proof of a phenotype. Keep unsupported mechanisms uncertain. Do not infer orthology, splicing effects, variant causality or calibrated confidence from these inputs alone. See the pipeline's `references/interpretation.md` for claim review.

Deliver output paths, actual completion/failure states, raw annotation/model responses, versions and hashes, and a concise scientific interpretation. Preserve original model outputs. Do not call mocked tests a real biological replication or claim identical new model responses from identical prompts. Do not publish files or send results to other people unless the user asks.
