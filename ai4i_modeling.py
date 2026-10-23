"""Train and evaluate leakage-safe binary classifiers for the AI4I dataset."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from ai4i_data import DATA_PATH, DATASET_PAGE_URL, load_ai4i_dataset
from ai4i_preprocessing import AI4IPreparedData, prepare_ai4i_data


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "ai4i_modeling" / "model_comparison.json"
POSITIVE_CLASS = 1
DEFAULT_THRESHOLD = 0.5


def evaluate_binary_predictions(
    y_true: Any,
    y_scores: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    positive_class: int = POSITIVE_CLASS,
) -> dict[str, Any]:
    """Return thresholded metrics and ranking metrics for binary probabilities.

    ``y_scores`` must contain probabilities for ``positive_class``. The
    confusion matrix always uses label order ``[0, 1]`` so its rows are true
    labels and columns are predicted labels in a stable, documented order.
    """
    truth = np.asarray(y_true)
    scores = np.asarray(y_scores, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_scores must be one-dimensional")
    if truth.size == 0 or truth.size != scores.size:
        raise ValueError("y_true and y_scores must be non-empty and have equal lengths")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("y_scores must contain finite probabilities in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value in [0, 1]")
    if positive_class not in (0, 1):
        raise ValueError("positive_class must be 0 or 1")
    observed_labels = set(np.unique(truth).tolist())
    if not observed_labels.issubset({0, 1}) or not observed_labels:
        raise ValueError("y_true must contain only binary labels 0 and 1")

    negative_class = 1 - positive_class
    predictions = np.where(scores >= threshold, positive_class, negative_class)
    binary_truth = (truth == positive_class).astype(int)
    confusion = confusion_matrix(truth, predictions, labels=[0, 1])
    has_both_labels = set(np.unique(binary_truth).tolist()) == {0, 1}

    return {
        "positive_class": int(positive_class),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(
            precision_score(
                truth,
                predictions,
                pos_label=positive_class,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(truth, predictions, pos_label=positive_class, zero_division=0)
        ),
        "f1": float(
            f1_score(truth, predictions, pos_label=positive_class, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(binary_truth, scores)) if has_both_labels else None
        ),
        # Average precision summarizes the precision-recall curve and is
        # reported as PR-AUC in this experiment.
        "pr_auc": (
            float(average_precision_score(binary_truth, scores))
            if int(binary_truth.sum()) > 0
            else None
        ),
        "confusion_labels": [0, 1],
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def _json_safe(value: Any) -> Any:
    """Convert estimator parameters and numpy scalars to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _class_counts(labels: Any) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _fit_and_evaluate(
    model: LogisticRegression | MLPClassifier,
    prepared: AI4IPreparedData,
) -> dict[str, Any]:
    """Fit an estimator on the shared training split and score the held-out set."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(prepared.X_train, prepared.y_train)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in captured
    )

    positive_index = list(model.classes_).index(POSITIVE_CLASS)
    positive_scores = model.predict_proba(prepared.X_test)[:, positive_index]
    return {
        "model_params": _json_safe(model.get_params(deep=False)),
        "training": {
            "iterations": _json_safe(getattr(model, "n_iter_", None)),
            "convergence_warning": bool(convergence_warning),
        },
        "metrics": evaluate_binary_predictions(
            prepared.y_test,
            positive_scores,
            threshold=DEFAULT_THRESHOLD,
            positive_class=POSITIVE_CLASS,
        ),
    }


def run_ai4i_modeling(
    data: Any,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compare Logistic Regression and MLP on one held-out stratified split.

    Both estimators use the same train-fitted preprocessing and split returned
    by :func:`prepare_ai4i_data`. The test partition is used only for final
    evaluation. No threshold or hyperparameter tuning is performed.
    """
    prepared = prepare_ai4i_data(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            penalty="l2",
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    }

    report = {
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "source": DATASET_PAGE_URL,
            "rows": int(len(data)),
        },
        "metadata": {
            "target": "Machine failure",
            "positive_class": POSITIVE_CLASS,
            "decision_threshold": DEFAULT_THRESHOLD,
            "metric_definitions": {
                "pr_auc": (
                    "Average precision, a summary of the precision-recall curve; "
                    "reported as PR-AUC."
                )
            },
            "confusion_matrix_layout": "rows=true [0,1], columns=predicted [0,1]",
            "random_state": int(random_state),
            "split": {
                "method": "stratified train/test split",
                "test_size": float(test_size),
                "train_samples": int(len(prepared.y_train)),
                "test_samples": int(len(prepared.y_test)),
                "stratified": True,
                "train_class_counts": _class_counts(prepared.y_train),
                "test_class_counts": _class_counts(prepared.y_test),
            },
            "feature_names": list(prepared.feature_names),
            "preprocessing": (
                "Type one-hot encoding and sensor standardization; both transformers "
                "fit on training rows only. IDs and failure-mode labels are excluded."
            ),
            "imbalance_note": (
                "Machine failure is the minority class. Accuracy alone can hide missed "
                "failures, so precision, recall, F1, ROC-AUC, PR-AUC and the confusion "
                "matrix are reported. Models use fixed settings without resampling or "
                "test-set threshold tuning."
            ),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        "models": {
            name: _fit_and_evaluate(model, prepared)
            for name, model in models.items()
        },
    }

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to AI4I CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the machine-readable model comparison JSON",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    report = run_ai4i_modeling(
        load_ai4i_dataset(args.data),
        test_size=args.test_size,
        random_state=args.random_state,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Model comparison: {args.output}")


if __name__ == "__main__":
    main()
