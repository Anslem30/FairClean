import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from fairclean.diagnostics import sensitive_attribute_flip
from fairclean.metrics import demographic_parity
from fairclean.mitigation import hybrid_mitigation


def test_demographic_parity():
    df = pd.DataFrame({"s": [0, 0, 1, 1]})
    result = demographic_parity(df, np.array([0, 1, 1, 1]), "s")
    assert result["rates"] == {0: 0.5, 1: 1.0}
    assert result["gap"] == 0.5
    assert result["ratio"] == 0.5


def test_sensitive_flip_detects_changes():
    df = pd.DataFrame({"s": [0, 1, 0, 1], "x": [0, 0, 1, 1]})
    model = DecisionTreeClassifier(random_state=42).fit(df, df["s"])
    changed = sensitive_attribute_flip(df, model, ["s", "x"], "s")
    assert changed == [0, 1, 2, 3]


def test_mitigation_keeps_row_identity():
    df = pd.DataFrame({"x": [10, 20, 30, 40]})
    cleaned, removed, reweighted, log = hybrid_mitigation(df, [1], [1], [2, 3])
    assert removed == 1
    assert reweighted == 2
    assert cleaned.loc[2, "weight"] == 0.5
    assert cleaned.loc[3, "weight"] == 0.5
    assert log["reweighted"].sum() == 2
