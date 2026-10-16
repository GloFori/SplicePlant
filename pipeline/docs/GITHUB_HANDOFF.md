# Integration into SplicePlant

Target repository: `GloFori/SplicePlant`, branch `pre-pipline`. The actual existing directory is **pipeline**, not `pipline`. All runtime code and both user interfaces belong under it.

## Intended tree

```text
SplicePlant/
  README.md                        # existing project overview; add a link below
  pipeline/
    README.md                      # human workflow first; optional Skill afterward
    0_install_environment.py
    0_download_databases.py
    1_afdb_seq_search_download.py
    2_af3_cif_to_pdb.py
    3_map_af3_pdbs.py
    4_cotton.py
    llm.py
    environment.local.yml
    plant_protein_trait/            # shared engine; required by all seven scripts
    skill/
      SKILL.md
      README.md
      scripts/ppt.py
    references/
    docs/
    tests/
    examples/README.md
```

The delivered ZIP contains this `pipeline/` directory. Extract it first. Do not upload the ZIP as the only repository file, put a second `pipeline/` inside the existing one, or copy only the seven thin entry scripts. Preserve the existing repository's other directories.

## Upload using the GitHub website

1. Save the existing branch commit identifier. The new package replaces the five historical files in `pipeline/` with maintained versions; the old source remains in Git history.
2. Return to the repository **root** on branch `pre-pipline`.
3. Select **Add file → Upload files**. Drag the extracted **pipeline folder** into the upload area. Check that paths begin `pipeline/`, not `pipeline/pipeline/`.
4. Inspect the list, including `pipeline/README.md`, both setup scripts, the shared engine and `pipeline/skill/SKILL.md`. Commit to `pre-pipline` with a message such as `Add manual protein-trait workflow and optional skill`.
5. Add a link to `pipeline/README.md` from the existing root README, without replacing the project overview.

Suggested root README addition:

```markdown
## Protein function and trait analysis

See [the pipeline guide](pipeline/README.md) for software/database installation,
manual reproduction, and the optional AI Skill.
```

## Upload using Git

Clone or open a clean checkout of `pre-pipline`. Copy the **contents** of the extracted `pipeline/` into the checkout's existing `pipeline/`, preserving that directory name. Review existing local modifications before replacing files. Then run from the repository root:

```bash
git diff --stat
git add pipeline
git diff --cached --stat
git diff --cached
git commit -m "Add manual protein-trait workflow and optional skill"
git push origin pre-pipline
```

Stage any separately edited root README explicitly. This package has not pushed, merged or opened a PR for you.

## Old entry points

The older root `plant-protein-trait/` package and `pipline_skill` placeholder are superseded by the new `pipeline/` guide. After verifying the new branch, remove or clearly mark those old entry points as deprecated in a separate reviewed commit. Do not delete server installation directories, resources or unfinished runs. The new engine accepts completed v0.2 resource manifests; see [manual migration](../references/manual.md).

## Verification after upload

Clone the updated branch into a fresh checkout, enter `pipeline/`, and follow the README's software tests and synthetic smoke test. Then perform the actual Linux installation and one real protein/model run. Retain results and version manifests. The supplied [validation record](VALIDATION.md) distinguishes checks already performed from those still requiring the reviewer's server and credentials.

An optional [CI workflow template](ci-example.yml) can be copied to the repository root at `.github/workflows/plant-protein-trait.yml`. It performs software tests only, with no databases or API secrets. Review an existing workflow of the same name before replacing it.

After a real clean-server run and scientific review, publish a tag/release according to the repository's conventions and record its exact commit in the manuscript. Add the real Arabidopsis demo when its input and archived outputs are ready; the synthetic smoke fixture is not that demo.
