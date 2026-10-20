# Scientific interpretation after freezing

Keep four distinct questions separate:

1. Targeted verify: given the nominated trait, does independent protein evidence support a plausible relationship?
2. Targeted explore: what testable mechanisms might connect that function to the nominated trait?
3. Blind verify: without that prior, which processes and traits are most defensible?
4. Blind explore: which additional evidence-linked hypotheses arise without that prior?

These are four model analyses, not four independent experiments. Shared training data and shared input annotations create dependence.

## Grade each mechanistic link

| Grade | Appropriate use |
| --- | --- |
| Direct evidence | A direct experiment supporting the particular claim on the actual target; not merely a computational family match |
| Indirect mechanistic evidence | Characterized homologs, conserved domains and sufficiently supported functional processes suggest a route |
| Biological plausibility | A reasonable process-to-trait connection remains untested for this target |
| Speculative connection | One or more essential steps lack specific support |

Separate certainty about protein family, catalytic activity, biological process, pathway, organ/tissue relevance and organismal phenotype. A high-quality fold or strong homology does not assign identical confidence to all these levels. Positive effect direction must come from actual evidence, not from the sign of an unrelated association.

## Compare A versus B

- Both independently nominate the relevant direction: describe convergence of the two inference settings, while retaining mechanistic and experimental limitations.
- Targeted supports, blind does not nominate it: describe **targeted functional compatibility/support**, not independent trait prediction.
- Only explore supports it: describe a hypothesis for testing, not a verify result.
- Neither supports it: state unsupported or insufficient, distinguishing absence of evidence from contrary evidence.

Do not use simple keyword overlap as proof that blind inference predicted the trait. Inspect whether it nominated the specific trait, a generic upstream process, or merely mentioned it in a limitation. Predefine synonyms or scoring criteria for any formal benchmark and report all runs.

## Post-freeze study context

Independent association, linkage, splicing-model and population observations may be discussed only after freezing. Clearly label user-supplied statistics whose raw data were not reanalyzed. Show which links are measured and which remain inferred:

```text
associated locus → candidate regulatory event → molecular consequence
                 → gene/protein function → relevant biological process → trait
```

Protein evidence alone cannot establish altered transcript abundance, altered splicing, a new isoform sequence or changed protein function. Those require independent observations. Association, LD and prediction scores do not establish causality or distinguish a causal variant from correlated alternatives.

Choose the strongest supported wording, not the desired wording:

1. Protein function only.
2. Trait-related biological process.
3. Plausible association with the target trait.
4. Population-supported **candidate mechanism** linking a molecular event to the trait, only when the independent population evidence and functional bridge support that description.

The fourth level is not causal proof. Reserve “causal”, “validated functional variant”, “controls” or “determines” for appropriate experimental causal evidence.

## Report contents

Include QC and source checks; homology statistics; domain and GO provenance; structure availability; all four analyses; A/B comparison; evidence table; post-freeze context; scientific limitations; proposed tests; and manuscript wording matched to the evidence. Keep original responses unchanged and put the reviewer's corrections in a separate interpretation file.
