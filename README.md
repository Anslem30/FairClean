FairClean Workflow: From Baseline Training to Final Fairness Evaluation

FairClean consists of a structured pipeline that performs fairness diagnostics, applies targeted data‑level mitigation, and outputs a fairness‑aware cleaned dataset for downstream model training. The workflow is summarised below.

1. Load and Preprocess the Dataset

-Import the raw dataset

-handle missing values, enecode categorical variable and normalise/scale numerical features

-Split into training and test sets

2. Train the Baseline Model

-Fit the baseline classifier

-Generate baseline predictions on the test set

3. Compute Baseline Fairness Metrics
Evaluate the baseline model using the three fairness metrics.

-Demogrpahic Parity Ratio(DPR)

-Demogrpahic Parity Difference(DPD)

-Conterfactual Sensitivity Rate(CSR)

These serve as the reference point for assessing fairness improvements.

4. Run FairCLean Diagnostic Tests

Apply the three diagnostic components to identify unfair or unstable samples(rows):

Attribute Flip Test

Checks whether the model’s prediction changes when the sensitive attribute (e.g., gender) is flipped while all other features remain fixed.

Causal Counterfactual Test

Generates a causally consistent counterfactual for each row by flipping the sensitive attribute and adjusting any causally linked features, then checks whether the model’s prediction changes.

Structural Uplift Test

Constructs an improved version of each row by setting key socio‑economic or qualification‑related features to their best‑case values, then checks whether the model’s prediction behaves as expected.

Each diagnostic outputs a binary flag per row indicating whether a fairness or stability violation occurred.

5. Classify Rows Based on Diagnostic Outcomes

Each row is assigned to one of three categories:

-Directly Unfair — fails the Attribute Flip Test

-Causally Unfair — fails the Causal Counterfactual Test

-Structurally Unstable — fails the Structural Uplift Test

This classification determines the mitigation applied to each row

6. Apply Hybrid Mitigation Strategy

FairClean applies a hybrid data‑level mitigation approach:

-Remove rows flagged as directly unfair

-Remove rows flagged as causally unfair

-Reweight rows flagged as structurally unstable

This produces the cleaned and reweighted training dataset.

7. Retrain the Model on the Cleaned Dataset

-Retrain the same model using the cleaned dataset and sample weights.

-Generate new predictions on the test set.

8. Compute Post‑Cleaning Fairness Metrics
Recompute:

-DPR

-DPD

-CSR

These values quantify the fairness improvements introduced by FairClean.

9. Generate Transparency Log

Produce a structured log containing:

-number of rows removed

-number of rows reweighted

-baseline fairness metrics

-post‑cleaning fairness metrics

-mitigation summary

This ensures full transparency and reproducibility of the FairClean process.

10. Final Output

The pipeline outputs:

-baseline fairness metrics

-post‑cleaning fairness metrics

-cleaned + reweighted dataset

-transparency log

This completes the full FairClean workflow from raw data to fairness‑aware evaluation.
