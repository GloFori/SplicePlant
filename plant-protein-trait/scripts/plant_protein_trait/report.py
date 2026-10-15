"""Compare frozen results without adding biological claims or editing model outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import fresh_dir, manifest, read_json, verify_manifest, write_json, write_tsv


def summarize(prepared: Path, output: Path, background: Path | None = None) -> Path:
    frozen = read_json(prepared / "results_frozen.json")
    verify_manifest(prepared, frozen)
    if frozen.get("status") != "four_results_frozen":
        raise ValueError("A completed four-result freeze is required")
    out = fresh_dir(output)
    packet = read_json(prepared / "protein_evidence.json")
    results = {f"{a}_{m}": read_json(prepared / f"{a}_{m}" / "result.json")
               for a in ("targeted", "blind") for m in ("verify", "explore")}
    assessments = {x["evidence_id"]: x for x in results["targeted_verify"]["evidence_assessment"]}
    rows = []
    for evidence in packet["evidence"]:
        assessment = assessments.get(evidence["evidence_id"], {})
        rows.append({"Evidence": evidence["evidence_id"] + ": " + evidence["kind"],
                     "Result": json.dumps(evidence["observation"], ensure_ascii=False),
                     "Supports target trait?": assessment.get("supports", "not assessed by model"),
                     "Strength": assessment.get("grade", "not assessed"),
                     "Interpretation": assessment.get("limitation", "requires scientific review")})
    write_tsv(out / "evidence_table.tsv", rows, ["Evidence", "Result", "Supports target trait?", "Strength", "Interpretation"])
    text = ["# Targeted versus blind comparison", "",
            "This comparison reproduces the four frozen model outputs. It is not independent biological validation.", "",
            "- Targeted: Given the target trait, does the protein evidence support a plausible relationship?",
            "- Blind: Without a target trait, which processes and phenotypes does the evidence suggest?", ""]
    for name, data in results.items():
        text += [f"## {name}", "", "Verdict: " + data["verdict"], "", data["molecular_function"], ""]
        for row in data["top5"]:
            text.append(f"{row['rank']}. {row['biological_process']} → {row['potential_trait']} "
                        f"({row['confidence']}; {', '.join(row['supporting_evidence']) or 'no evidence'})")
        text += ["", "Strongest link: " + data["strongest_link"], "", "Weakest link: " + data["weakest_link"], ""]
    if background:
        contents = background.read_bytes()
        contents.decode("utf-8")  # fail on non-text instead of concealing content
        (out / "postfreeze_background.txt").write_bytes(contents)
        text += ["## Post-freeze context", "",
                 "Independent user-supplied context is stored in postfreeze_background.txt. "
                 "It was not included in any of the four inference prompts. Its factual accuracy is not verified by this tool.", ""]
    text += ["## Required scientific review", "",
             "1. Check each molecular-function claim against the raw annotation and aligned regions.",
             "2. Separate generic cellular processes from evidence specific to the proposed trait.",
             "3. State whether the blind outputs actually nominate the target or only a broad related process.",
             "4. Treat targeted-only support as functional compatibility, not independent trait prediction.",
             "5. Assign the claim level only after reviewing independent experimental/population evidence:",
             "   protein function; trait-related process; plausible trait association; or a population-supported candidate mechanism.",
             "6. Do not infer causality, effect direction, changed isoform function or validated variants from these outputs alone.",
             "7. Treat verify/explore as analysis styles, not independent experiments or calibrated confidence estimates.", ""]
    (out / "comparison.md").write_text("\n".join(text), encoding="utf-8")
    write_json(out / "report_manifest.json", manifest(out,
               [p.name for p in out.iterdir() if p.is_file()], source_results_manifest=frozen))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare frozen A/B outputs; optionally add background only after freezing")
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--background", type=Path)
    args = p.parse_args(argv)
    try:
        print(summarize(args.prepared, args.output, args.background))
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc)) from None
