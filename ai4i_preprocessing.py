"""Leakage-safe preprocessing and stratified splitting for AI4I classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Machine failure"
TYPE_COLUMN = "Type"
NUMERIC_FEATURE_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
FEATURE_COLUMNS = (TYPE_COLUMN, *NUMERIC_FEATURE_COLUMNS)
EXCLUDED_COLUMNS = (
    "UDI",
    "Product ID",
    TARGET_COLUMN,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)


@dataclass(frozen=True)
class AI4IPreparedData:
    """Raw and transformed partitions plus the fitted preprocessing pipeline."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]


def _make_preprocessor() -> ColumnTransformer:
    """Build a fresh transformer; callers fit it only on the training partition."""
    return ColumnTransformer(
        transformers=[
            (
                "type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [TYPE_COLUMN],
            ),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def prepare_ai4i_data(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> AI4IPreparedData:
    """Split AI4I data and fit categorical/numeric transforms on training data only.

    The target is ``Machine failure``. The six model inputs are ``Type`` and the
    five sensor measurements; IDs, the target, and subtype failure labels are
    excluded to prevent identifier artifacts and target leakage.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    required_columns = (*FEATURE_COLUMNS, TARGET_COLUMN)
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required AI4I columns: {missing_columns!r}")

    features = data.loc[:, FEATURE_COLUMNS].copy()
    target = data.loc[:, TARGET_COLUMN].copy()
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Model feature columns contain missing values: {missing!r}")
    if target.isna().any():
        raise ValueError(f"Target {TARGET_COLUMN!r} contains missing values")

    labels = set(target.unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(
            f"Target {TARGET_COLUMN!r} must contain both binary classes 0 and 1; "
            f"got {sorted(labels)!r}"
        )

    numeric_columns = list(NUMERIC_FEATURE_COLUMNS)
    non_numeric = [
        column for column in numeric_columns if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise ValueError(f"Numeric model features have non-numeric values: {non_numeric!r}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    preprocessor = _make_preprocessor()
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=float)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)
    feature_names = tuple(str(name) for name in preprocessor.get_feature_names_out())

    return AI4IPreparedData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
