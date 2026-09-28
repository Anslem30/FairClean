"""Counterfactual diagnostics extracted from the original FairClean notebooks."""

import numpy as np


def _predict(model, X, scaler=None):
    return model.predict(scaler.transform(X) if scaler is not None else X)


def sensitive_attribute_flip(df, model, feature_cols, sensitive_col, scaler=None):
    """Return row positions whose prediction changes when a binary sensitive attribute is flipped."""
    X = df[list(feature_cols)].copy()
    factual = _predict(model, X, scaler)
    X_cf = X.copy()
    X_cf[sensitive_col] = 1 - X_cf[sensitive_col]
    counterfactual = _predict(model, X_cf, scaler)
    return np.flatnonzero(factual != counterfactual).tolist()


def structural_uplift(df, model, feature_cols, dataset, scaler=None):
    """Apply the dataset-specific structural uplift used in the submitted project notebooks."""
    X = df[list(feature_cols)].copy()
    factual = _predict(model, X, scaler)
    X_cf = X.copy()

    if dataset == "adult":
        for col in ["education-num", "capital-gain", "capital-loss", "hours-per-week"]:
            if col in X_cf.columns:
                X_cf[col] = X_cf[col].max()
        for prefix in ["occupation_", "workclass_", "marital-status_"]:
            cols = [c for c in X_cf.columns if c.startswith(prefix)]
            if cols:
                X_cf[cols] = 0
                X_cf[cols[0]] = 1
    elif dataset == "graduates":
        for prefix, chosen in [
            ("Education_Level_", -1),
            ("University_Ranking_", 0),
            ("Language_Proficiency_", -1),
            ("Job_Sector_", 0),
        ]:
            cols = [c for c in X_cf.columns if c.startswith(prefix)]
            if cols:
                X_cf[cols] = 0
                X_cf[cols[chosen]] = 1
        if "internship_bin" in X_cf.columns:
            X_cf["internship_bin"] = 1
        if "GPA" in X_cf.columns:
            X_cf["GPA"] = 1.0
        if "Salary" in X_cf.columns:
            X_cf["Salary"] = 1.0
    else:
        raise ValueError("dataset must be 'adult' or 'graduates'")

    counterfactual = _predict(model, X_cf, scaler)
    return np.flatnonzero(factual != counterfactual).tolist()


def causal_counterfactual_prototype(df, model, feature_cols, sensitive_col, scaler=None):
    """Reproduce the submitted prototype's diagnostic labelled as causal.

    In the submitted notebooks this is operationally identical to the direct
    sensitive-attribute flip. It is retained for reproducibility and should
    not be interpreted as an independent causal model.
    """
    return sensitive_attribute_flip(df, model, feature_cols, sensitive_col, scaler)
