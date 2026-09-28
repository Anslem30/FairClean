# Methodology

## Research question

FairClean investigates whether counterfactual reasoning can move beyond fairness diagnosis and guide targeted preprocessing interventions.

## Diagnostic layers

### 1. Sensitive-attribute flip

The protected attribute is changed while the remaining feature values are held constant. A changed prediction identifies direct model sensitivity to that attribute.

### 2. Prototype causal check

The submitted research design distinguishes a causal-counterfactual diagnostic from the direct attribute flip. However, in the submitted notebooks reviewed for this portfolio refactor, the implemented causal helper mirrors the direct sensitive-attribute flip rather than independently modifying downstream causal features.

The refactored package therefore preserves that behaviour for reproducibility but labels it explicitly as a **prototype causal check**, not an independent causal model.

### 3. Structural uplift

Selected non-sensitive socioeconomic features are moved toward favourable values. Prediction changes under this test are treated differently from direct sensitivity: affected rows are reweighted rather than automatically removed.

## Hybrid mitigation

FairClean combines two interventions:

- **Removal:** rows identified by the direct/prototype-causal diagnostics are removed from the cleaned training data.
- **Reweighting:** structurally unstable rows that remain are assigned a lower training weight.

A row-level cleaning log records diagnostic flags and the intervention applied.

## Evaluation

The prototype was evaluated with Logistic Regression and Decision Tree classifiers on the Adult Income and International Graduates datasets. Group-level demographic-parity measures were compared before and after mitigation, alongside counts from the diagnostics.

The reported results are model-dependent. Adult Logistic Regression improved its demographic-parity gap, while Adult Decision Tree did not. The International Graduates experiments showed almost no baseline demographic-parity disparity.

## Interpretation

FairClean is an academic research prototype, not a production fairness or compliance system. Counterfactual fairness depends on causal assumptions, and dataset-specific uplift rules should not be interpreted as universal definitions of fairness.
