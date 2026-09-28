"""Fairness metrics used in the FairClean experiments."""

import numpy as np
import pandas as pd


def demographic_parity(df, predictions, sensitive_col):
    """Return group positive rates, signed gap (group 1 - group 0), and ratio."""
    tmp = pd.DataFrame({sensitive_col: df[sensitive_col].to_numpy(), "pred": predictions})
    rates = tmp.groupby(sensitive_col)["pred"].mean().to_dict()
    r0, r1 = rates.get(0, np.nan), rates.get(1, np.nan)
    gap = r1 - r0
    ratio = r0 / r1 if pd.notna(r1) and r1 > 0 else np.nan
    return {"rates": rates, "gap": gap, "ratio": ratio}


def adult_extended_metrics(df, predictions, sensitive_col="sex_bin", target_col="income"):
    """Reproduce the extended Adult Income metrics used in the Decision Tree notebook."""
    tmp = df[[sensitive_col, target_col]].copy()
    tmp["pred"] = predictions
    group = tmp.groupby(sensitive_col)
    positive_rate = group["pred"].mean()
    tpr = group.apply(lambda g: ((g["pred"] == 1) & (g[target_col] == 1)).sum() /
                                 max((g[target_col] == 1).sum(), 1))
    fpr = group.apply(lambda g: ((g["pred"] == 1) & (g[target_col] == 0)).sum() /
                                 max((g[target_col] == 0).sum(), 1))
    return {
        "dp_gap": positive_rate[1] - positive_rate[0],
        "dp_ratio": positive_rate[0] / positive_rate[1] if positive_rate[1] > 0 else np.nan,
        "eo_gap": tpr[1] - tpr[0],
        "fpr_gap": fpr[1] - fpr[0],
        "rates": positive_rate.to_dict(),
    }
