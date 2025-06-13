from __future__ import annotations

"""Data loading and preprocessing utilities.

For demo purposes we use the builtin Iris dataset from scikit-learn, but you can
swap this out for any custom loader later.  The helper returns Pandas
DataFrames / Series which play nicely with the rest of the pipeline.
"""

from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

__all__ = [
    "load_demo_iris",
]


def load_demo_iris(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the Iris dataset and split train/val.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, y_train, X_test, y_test
