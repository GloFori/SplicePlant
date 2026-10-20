# Validation record for v0.3.0

Date: 2026-10-15. This record describes software validation; it is not a completed biological replication.

## Executed checks

| Check | Environment | Result |
| --- | --- | --- |
| 74 automated tests | Windows, Python 3.12.14, Biopython 1.85 | All passed; no skipped tests |
| Seven public script interfaces and subcommands, setup plan, Skill location and smoke execution | Windows, Python 3.14.4 | 19 CLI checks passed |
| Synthetic offline smoke test | Windows, Python 3.14.4 | QC, annotation interface, four isolated prompts and hashes passed |
| Skill frontmatter validator | Bundled skill-creator validator; Python 3.12 UTF-8 mode | Passed |
| Python syntax compatibility | Python 3.10 grammar | 30 source files parsed |
| Relative Markdown links | Local source package | All checked links resolve |
| Private input scan | Source, docs, tests | No user contact, private input sequence/ID, server path or live job ID bundled |

The 52 inherited tests cover parsers, sequence binding, strict protein QC, failure preservation, targeted/blind isolation, model response validation, evidence IDs, hashing/freezing, parameter alignment, remote polling and resumption, local failures without remote fallback, and full workflow execution with four mocked model calls.

The 22 added tests cover isolated installation plans, architecture rejection, install failure recording, interrupted-environment recovery, database checkpoint reuse after later failure, missing/changed resource detection, manifest-driven configuration, resource-snapshot archiving and resume binding, Skill installation/path resolution, Conda dispatch, and byte-identical human/installed-Skill prompt preparation. Structure checks include accession ranking, mock AFDB retrieval and cache reuse, reference-versus-target roles, explicit service failure, archive path/size checks, **actual Biopython CIF-to-PDB conversion**, filename collisions, isolated residues, full-chain matching, multiple-chain selection, and experimental B-factor handling. Structure fixtures are synthetic.

Installation commands, external annotation tools, model responses and online services are mocked in these tests. The tests do not download real annotation databases or make live model calls. Real local filesystem operations, subprocess calls to the supplied entry scripts, Skill copying/path binding and Biopython parsing/conversion are exercised.

The smoke test explicitly records biological annotation and model inference as `not_run`. It is not the deferred Arabidopsis demonstration. The real example and its four model outputs must be added separately.

## Reproduce software checks

From `pipeline/`, preferably in the supplied environment:

```bash
python -m unittest discover -s tests -v
python llm.py smoke-test --output /path/to/new/smoke_output
python llm.py pipeline --help
python 0_install_environment.py --root /path/to/dedicated_install --with-databases
python skill/scripts/ppt.py locate
```

The setup command above is plan-only. Add `--execute` only to perform installation. The five actual Biopython tests are explicitly skipped if Biopython is absent; use the supplied environment to run all 74 tests. Tests can use `PPT_TEST_TMP` when the system temporary directory is restricted.

## Still to validate on the publication server

- Linux Conda dependency resolution and actual execution of BLAST+, HMMER and InterProScan; full resource download, extraction and computation against the intended snapshot.
- Live NCBI/EBI/AFDB requests and actual local-versus-online evidence comparisons, with observed database/application releases preserved.
- Actual DashScope, compatible-provider or Ollama calls using the selected model and account settings. The tested protocol does not guarantee identical regenerated wording.
- The full clean-server demonstration and independent scientific review of its conclusions.
- Host-specific AI discovery/invocation. The installed Skill bridge was tested; no third-party AI host was configured during this task.
- GitHub Actions after the optional template is installed at the repository root. It is configured for Ubuntu and Python 3.12/3.13; no hosted run has occurred here.

No `pip install .` or wheel build is required by the documented source-directory workflow, and those distribution modes were not validated for this release. Keep the checkout with its supporting files. The local resource installer intentionally does not claim byte-identical historical databases: retain the completed resource snapshot and environment export. Raw responses and frozen manifests, not a future live rerun, preserve the original reported model outputs.
