# Portfolio refactor notes

This repository separates the **submitted research implementation** from later code-quality improvements.

## Original research record

The six notebooks in the repository root are preserved because they show the original preprocessing and FairClean experiments and retain the project's development history.

## Reusable implementation

The later `src/fairclean/` package extracts reusable logic for:

- demographic-parity metrics;
- direct sensitive-attribute counterfactual testing;
- dataset-specific structural uplift;
- hybrid removal/reweighting;
- model construction.

## Implementation details documented during refactoring

1. In the submitted notebooks, the helper labelled as the causal counterfactual test reproduces the direct sensitive-attribute flip rather than independently modelling downstream causal changes. The refactored code calls this `causal_counterfactual_prototype` and documents the limitation.
2. The reusable mitigation implementation keeps a stable `_fairclean_row_id` while removing and reweighting rows. This avoids ambiguity when structural row positions are mapped after removals.

The dissertation/notebook results reported in the README are retained as **reported research results**. They are not silently replaced by outputs from the later refactor.
