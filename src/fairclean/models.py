"""Model helpers corresponding to the two classifiers used in FairClean."""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def decision_tree(random_state=42):
    return DecisionTreeClassifier(
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        random_state=random_state
    )


def logistic_regression():
    return LogisticRegression(max_iter=500, class_weight="balanced")


def fit_logistic(model, X_train, y_train, sample_weight=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model.fit(X_scaled, y_train, sample_weight=sample_weight)
    return model, scaler
