"""Hybrid mitigation and transparent row-level logging."""

import pandas as pd


def hybrid_mitigation(df, direct_rows, causal_rows, structural_rows, structural_weight=0.5):
    """Remove direct/causal flagged rows and down-weight retained structural rows."""
    working = df.copy()
    working["_fairclean_row_id"] = range(len(working))
    remove_ids = set(direct_rows) | set(causal_rows)
    structural_ids = set(structural_rows)

    keep = ~working["_fairclean_row_id"].isin(remove_ids)
    cleaned = working.loc[keep].copy()
    cleaned["weight"] = 1.0
    reweight_mask = cleaned["_fairclean_row_id"].isin(structural_ids)
    cleaned.loc[reweight_mask, "weight"] = structural_weight

    log = pd.DataFrame({
        "row_index": range(len(working)),
        "attribute_unfair": [i in set(direct_rows) for i in range(len(working))],
        "causal_unfair": [i in set(causal_rows) for i in range(len(working))],
        "structural_unfair": [i in structural_ids for i in range(len(working))],
        "removed": [i in remove_ids for i in range(len(working))],
        "reweighted": [(i in structural_ids) and (i not in remove_ids) for i in range(len(working))],
    })

    cleaned = cleaned.set_index("_fairclean_row_id", drop=True)
    return cleaned, len(remove_ids), int(reweight_mask.sum()), log
