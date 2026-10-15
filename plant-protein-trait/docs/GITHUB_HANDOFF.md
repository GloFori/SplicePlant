# Add this project to your existing GitHub repository

This delivery is a local, reviewable source package. No repository was created and no remote files were changed. The package deliberately does not know your existing repository URL, license or author list.

## Recommended layout

Keep the complete folder under your existing repository:

```text
your-repository/
  .github/workflows/plant-protein-trait.yml
  skills/plant-protein-trait/
    SKILL.md
    README.md
    README.zh-CN.md
    agents/
    scripts/
    references/
    tests/
    docs/
    examples/
    pyproject.toml
```

Run commands from `skills/plant-protein-trait/`. Do not replace your repository's existing README or LICENSE. Add a link from its README to this folder.

The `.github/workflows/ci.yml` inside this package is a ready workflow, but GitHub executes workflows only from the **repository root's** `.github/workflows/`. Copy it there under the unique name `plant-protein-trait.yml`. It locates the package in `skills/plant-protein-trait/` or at the repository root and runs offline tests, smoke preparation and Python package installation. No secrets or paid model calls are needed. Do not overwrite an existing workflow of the same name without reviewing it.

## Before committing

From the existing repository root:

```bash
git status --short
```

After copying the files, test from the package directory:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --output runs/prepublish_smoke
```

Return to the repository root, inspect and commit only the intended files:

```bash
git add skills/plant-protein-trait .github/workflows/plant-protein-trait.yml
git diff --cached --stat
git diff --cached
git commit -m "Add reproducible single-protein trait analysis skill"
git push
```

These commands assume your existing branch/remote and Git authentication are already configured. They do not create a new repository, force-push, or delete prior files.

## Release and reviewer handoff

- Apply the existing repository's actual license and citation/author information.
- Check the actual GitHub Actions result after pushing; the local package does not claim hosted CI has already run.
- Add the real Arabidopsis example later according to `examples/README.md` and record a new version for that addition.
- Tag a reviewed version, for example `plant-protein-trait-v0.1.0`, using your repository's release conventions. Preserve the tag/commit in the manuscript methods and archive the corresponding data.
- Use an immutable release or an institutional/archive record for the final scientific reproducibility package. A changing branch URL alone is insufficient provenance.
- Reviewers can use the CLI without installing an AI skill or using a particular assistant. The local smoke test proves software plumbing only; the paper's real annotation and model artifacts must be archived separately.

## Skill installation for others

Install or copy the `skills/plant-protein-trait` folder as one unit into their skill host. For a GitHub-aware Skill installer, supply **your actual repository URL and this subdirectory path**, then pin the published revision where supported. Do not use a guessed owner/repository name.
