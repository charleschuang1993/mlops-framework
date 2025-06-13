from __future__ import annotations

"""Training utilities that fit a model and log artifacts to MLflow."""

from typing import Dict

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .data import load_demo_iris

__all__ = ["train_demo"]


def train_demo(
    C: float = 1.0,
    max_iter: int = 200,
    mlflow_tracking_uri: str | None = None,
) -> Dict[str, float]:
    """Train a logistic regression on the Iris dataset and log to MLflow.

    Parameters
    ----------
    C, max_iter : Logistic regression hyper-parameters.
    mlflow_tracking_uri : If provided, overrides env `MLFLOW_TRACKING_URI`.

    Returns
    -------
    metrics dict
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    X_train, y_train, X_test, y_test = load_demo_iris()

    with mlflow.start_run(run_name="logreg_demo"):
        model = LogisticRegression(C=C, max_iter=max_iter, multi_class="auto")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_micro": f1_score(y_test, preds, average="micro"),
        }

        mlflow.log_params({"C": C, "max_iter": max_iter})
        mlflow.log_metrics(metrics)
        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        return metrics
