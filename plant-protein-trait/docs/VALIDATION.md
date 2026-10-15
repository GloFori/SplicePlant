# Validation record for v0.1.0

Date: 2026-10-14. This is a software validation record, not a biological replication claim.

## Locally executed

| Check | Environment | Result |
| --- | --- | --- |
| 33 deterministic parser and protocol tests | Windows, Python 3.14.4 | All passed |
| The same 33 tests | Windows, Python 3.12.14 | All passed |
| Synthetic offline CLI smoke test | Windows, Python 3.14.4 | QC, annotation interface, four isolated prompts and input hashes passed |
| Skill frontmatter validator | Bundled skill-creator validator | Passed |
| Syntax and document integrity | Python 3.10 grammar; local documentation | 14 Python files, 21 local Markdown links and 2 YAML files passed |
| Python wheel build and isolated entry-point loading | Windows, Python 3.14.4 | Wheel source bytes match; all 3 CLI entry points return help |

The tests use synthetic sequences, XML/JSON fixtures and mocked network/model responses. They check sequence identity binding; BLAST identity and coordinate coverage; XML2 namespaces; nested InterPro GO extraction; Pfam independent domain E-values; legacy annotation loading; rejection of proxy/unverified structure scores; strict single-protein QC; blind-context checks; fresh stateless provider requests; failure and truncation preservation; evidence-ID validation; direct-evidence overstatement rejection; manifest tampering; and four-result freezing followed by comparison.

The smoke test generates a clearly labeled synthetic protein, records unavailable evidence as unavailable, and prepares prompts. It does **not** annotate a real protein or produce an LLM interpretation. The complete run/freeze/report path is tested with mocked responses, not a live model.

The managed local Windows host restricts Python's private temporary directories. The wheel was built through setuptools' underlying wheel command and checked in an isolated extracted directory. A normal `pip install .` was not completed locally; the supplied GitHub workflow includes that check. This host-specific build workaround is not required to run the source scripts.

## Reproduce these checks

From the project directory, using Python 3.10 or later:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --output runs/validation_smoke
```

Use a new smoke-test output directory for every attempt. On a managed Windows host where the default temporary directory is restricted, set `PPT_TEST_TMP` to a writable test directory. This does not change test assertions.

## Validation still to perform on the publication environment

- Live NCBI BLAST, EBI InterProScan and AlphaFold DB requests. API formats were reviewed against primary documentation, but no real sequence was submitted during this packaging task.
- Actual local BLAST+/HMMER executables and real database releases on the intended Linux server.
- Real Ollama, DashScope or compatible model calls, with the exact model version and provider settings recorded.
- A real Arabidopsis example, its raw annotations, four model outputs and scientific review. The author deferred this example; no gene or accession has been invented.
- Hosted GitHub Actions. The included workflow is configured for Ubuntu and Windows with Python 3.10 and 3.13, but has not run in the author's repository yet.

Online databases and hosted model aliases can change. A rerun is not guaranteed to produce identical annotations or wording. Archive the exact raw inputs, responses, prompts, versions and manifests used in a manuscript; preserve the corresponding code commit or release tag.

No publication-ready biological conclusion follows from passing these software checks. Schema validation cannot determine whether a proposed molecular mechanism is correct.
