# FairClean — Fairness-Aware Data Cleaning for Machine Learning

**Python · Pandas · NumPy · scikit-learn · Data Cleaning · Algorithmic Fairness · Responsible AI**

> **Portfolio highlight:** In the reported Adult Income Logistic Regression experiment, the demographic-parity gap decreased from **0.3308 to 0.2427** — approximately **26.6%**. Other experiments did not show the same improvement, highlighting that fairness interventions need to be evaluated rather than assumed to generalise.

FairClean is the framework I developed for my **BSc (Hons) Computer Science Honours project at Edinburgh Napier University**. It investigates whether fairness considerations can be introduced during data preparation, before patterns in training data become embedded in downstream machine-learning models.

The project combines fairness diagnostics, targeted data-level mitigation and transparent intervention logging. The repository preserves my original research notebooks and also contains a later portfolio refactor that extracts reusable Python components and tests.

**Project website:** https://www.fairclean.solutions/

## The problem

Traditional data cleaning focuses on missing values, inconsistent labels, duplicates and other technical data-quality problems. Those steps improve dataset quality, but they do not necessarily address discriminatory patterns already present in the data.

FairClean investigates:

> Can counterfactual reasoning be used not only to diagnose model sensitivity, but also to guide targeted data-cleaning interventions?

## How FairClean works

```text
Dataset
  ↓
Preprocessing and validation
  ↓
Baseline model
  ↓
┌─────────────────────────────────────┐
│ FairClean diagnostics               │
│ 1. Sensitive-attribute flip         │
│ 2. Prototype causal check*          │
│ 3. Structural uplift                │
└─────────────────────────────────────┘
  ↓
Hybrid mitigation
  ├─ Direct / prototype-causal flag → remove
  └─ Structural instability → reweight
  ↓
Retrain the same model
  ↓
Post-cleaning fairness evaluation
  ↓
Transparency log
```

***Implementation note:** The submitted research design distinguishes a causal-counterfactual test from the direct attribute-flip test. In the submitted notebooks currently in this repository, however, the implemented causal helper mirrors the direct sensitive-attribute flip rather than independently adjusting downstream causal features. The refactored package preserves and documents that behaviour rather than presenting it as a separate causal model.

## Experimental design

FairClean was evaluated across two employment-related datasets and two classifiers:

| Dataset | Decision Tree | Logistic Regression |
|---|:---:|:---:|
| Adult Income | ✓ | ✓ |
| International Graduates | ✓ | ✓ |

The workflow compares fairness measurements before and after mitigation and records the rows removed or reweighted.

## Reported results

| Dataset / model | DP gap before | DP gap after | Rows removed | Rows reweighted | Result |
|---|---:|---:|---:|---:|---|
| Adult Income — Logistic Regression | 0.3308 | **0.2427** | 6,087 | 27,586 | Gap reduced |
| Adult Income — Decision Tree | 0.1949 | 0.1969 | 706 | 36,564 | No improvement |
| International Graduates — Logistic Regression | 0.00045 | 0.00045 | 0 | 150,179 | Unchanged |
| International Graduates — Decision Tree | 0.00042 | 0.00042 | 0 | 150,184 | Unchanged |

The Adult Logistic Regression result corresponds to an approximately **26.6% reduction in demographic-parity gap**. Importantly, the Decision Tree result did not improve on the same measure. FairClean is therefore presented as an experimental framework whose effects are **model- and dataset-dependent**, not as a method guaranteed to improve fairness.

The machine-readable results summary is available in `results/metrics/fairclean_results_summary.csv`.

## Original FairClean workflow

### 1. Load and preprocess the dataset
- Import the dataset.
- Handle missing values.
- Encode categorical variables and normalise or scale numerical features.
- Split the data into training and test sets.

### 2. Train the baseline model
- Fit the baseline classifier.
- Generate baseline predictions on the test set.

### 3. Compute baseline fairness metrics
The original experiments evaluate measures including:
- Demographic Parity Ratio (DPR)
- Demographic Parity Difference / gap (DPD)
- Counterfactual sensitivity diagnostics

These provide the reference point for post-cleaning evaluation.

### 4. Run FairClean diagnostics

**Sensitive-attribute flip**  
Flip the sensitive attribute while keeping the remaining features fixed and identify predictions that change.

**Prototype causal check**  
Retained to reproduce the submitted implementation. See the implementation note above.

**Structural uplift**  
Move selected socioeconomic or qualification-related features toward favourable values and identify predictions that change.

### 5. Classify diagnostic outcomes
Rows are flagged according to direct/prototype-causal sensitivity and structural instability.

### 6. Apply hybrid mitigation
- Remove rows flagged by the direct/prototype-causal diagnostics.
- Reweight retained rows flagged as structurally unstable.
- Record the intervention in a transparency log.

### 7. Retrain the model
Retrain the same classifier using the cleaned dataset and sample weights.

### 8. Re-evaluate fairness
Recompute the relevant fairness measurements on the retrained model.

### 9. Generate the transparency log
Record diagnostic flags, removals and reweighting so interventions can be inspected.

### 10. Produce the final research outputs
The experiments produce before/after fairness measurements, intervention counts and the cleaned/reweighted training data used for retraining.

## Repository structure

```text
FairClean/
├── Adult-income1.ipynb              # Original Adult preprocessing
├── Adult-income2.ipynb              # Original Adult preprocessing/analysis
├── FairCleanAdultModel1.ipynb       # Original Adult Decision Tree experiment
├── FairCleanAdultModel2.ipynb       # Original Adult Logistic Regression experiment
├── FairCleanGradModel1.ipynb        # Original Graduates Decision Tree experiment
├── FairCleanGradModel2.ipynb        # Original Graduates Logistic Regression experiment
├── adult-income-preprocessed.csv    # Processed Adult data used in the project
├── src/
│   └── fairclean/                    # Reusable portfolio refactor
├── tests/                            # Unit tests for reusable components
├── results/
│   └── metrics/                      # Reported experiment summary
├── data/
│   └── README.md                     # Dataset attribution/licensing notes
├── docs/
│   ├── methodology.md
│   └── refactor_notes.md
├── pyproject.toml
├── requirements.txt
└── CITATION.cff
```

## Running the reusable package

Clone the repository, create a Python environment, then install the project in editable mode:

```bash
pip install -e .[dev]
pytest -q
```

The original notebooks are retained as the academic research record. Some were developed using Google Drive/Colab paths, so they are not presented as the canonical package interface. The reusable components under `src/fairclean/` are the cleaner entry point for inspecting the later refactor.

## Data attribution

The Adult Income data used in the project originates from the **UCI Machine Learning Repository**. Attribution and licensing notes are documented in `data/README.md`.

The International Graduates dataset used in the research is **not redistributed here** because its original distribution licence has not yet been verified confidently enough for public redistribution.

## Limitations

FairClean is an **academic research prototype**, not a production fairness, hiring, visa, lending or compliance system.

Key limitations include:

- fairness depends on context and cannot be established by demographic parity alone;
- structural-uplift rules are dataset-specific;
- the submitted causal helper does not independently implement the full causal-counterfactual mechanism described in the research design;
- the reported mitigation effects differ across models and datasets;
- the later reusable-code refactor should be distinguished from the original submitted experiment notebooks.

See `docs/refactor_notes.md` for the implementation details identified during portfolio refactoring.

## Technologies

**Python · Pandas · NumPy · scikit-learn · Jupyter Notebook · Machine Learning · Data Cleaning · Model Evaluation · Algorithmic Fairness · Responsible AI**

## Author

**Uyi Anslem Ezama**  
First-Class BSc (Hons) Computer Science  
Edinburgh Napier University

[LinkedIn](https://www.linkedin.com/in/uyi-ezama-b883a0331/) · [GitHub](https://github.com/Anslem30) · [FairClean website](https://www.fairclean.solutions/)
