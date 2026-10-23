"""Create reproducible AI4I figures from the saved model evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from ai4i_data import DATA_PATH, load_ai4i_dataset
from ai4i_modeling import (
    DEFAULT_OUTPUT_PATH as DEFAULT_MODEL_RESULTS_PATH,
    dataframe_sha256,
    evaluate_binary_predictions,
)
from ai4i_preprocessing import (
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_ai4i_data,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "ai4i"
MODEL_NAMES = ("logistic_regression", "mlp")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
}


def _validate_report(data: pd.DataFrame, report: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Check saved results match this dataset and reproduce their test split."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(report, dict) or "metadata" not in report or "models" not in report:
        raise ValueError("model_results must be a model comparison report")
    dataset_metadata = report.get("dataset", {})
    if int(dataset_metadata.get("rows", -1)) != len(data):
        raise ValueError("model report row count does not match the supplied dataset")
    expected_fingerprint = dataset_metadata.get("data_sha256")
    if not expected_fingerprint or dataframe_sha256(data) != expected_fingerprint:
        raise ValueError("model report data fingerprint does not match the supplied dataset")

    metadata = report["metadata"]
    if int(metadata.get("positive_class", -1)) != 1:
        raise ValueError("AI4I visualizations expect Machine failure=1 as positive class")
    split = metadata.get("split", {})
    if not split.get("stratified"):
        raise ValueError("model report must record a stratified train/test split")
    prepared = prepare_ai4i_data(
        data,
        test_size=float(split["test_size"]),
        random_state=int(metadata["random_state"]),
    )
    if len(prepared.y_test) != int(split.get("test_samples", -1)):
        raise ValueError("model report test size does not match the reproduced split")

    reference_y = np.asarray(prepared.y_test, dtype=int)
    for model_name in MODEL_NAMES:
        if model_name not in report["models"]:
            raise ValueError(f"model report is missing {model_name!r}")
        model_result = report["models"][model_name]
        evaluation_data = model_result.get("evaluation_data", {})
        y_true = np.asarray(evaluation_data.get("y_true", []), dtype=int)
        scores = np.asarray(evaluation_data.get("positive_scores", []), dtype=float)
        if y_true.shape != reference_y.shape or not np.array_equal(y_true, reference_y):
            raise ValueError(
                f"saved {model_name} evaluation labels do not match the dataset split"
            )
        if scores.shape != reference_y.shape or not np.isfinite(scores).all():
            raise ValueError(f"saved {model_name} evaluation scores are missing or invalid")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"saved {model_name} probabilities must be in [0, 1]")
        metrics = model_result.get("metrics", {})
        expected_metrics = evaluate_binary_predictions(
            y_true,
            scores,
            threshold=float(metrics.get("threshold", -1)),
            positive_class=int(metrics.get("positive_class", -1)),
        )
        if metrics.get("confusion_labels") != expected_metrics["confusion_labels"]:
            raise ValueError(f"saved {model_name} confusion-matrix labels do not match")
        if metrics.get("confusion_matrix") != expected_metrics["confusion_matrix"]:
            raise ValueError(f"saved {model_name} confusion matrix does not match its scores")
        for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            saved_value = metrics.get(metric_name)
            expected_value = expected_metrics[metric_name]
            if expected_value is None:
                matches = saved_value is None
            else:
                matches = saved_value is not None and np.isclose(
                    float(saved_value),
                    float(expected_value),
                    rtol=1e-10,
                    atol=1e-12,
                )
            if not matches:
                raise ValueError(
                    f"saved {model_name} metrics do not match its y_true and scores"
                )
    return prepared, metadata


def _save_confusion_matrices(report: dict[str, Any], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    class_names = ["No failure (0)", "Failure (1)"]
    for ax, model_name in zip(axes, MODEL_NAMES):
        model_result = report["models"][model_name]
        metrics = model_result["metrics"]
        if metrics.get("confusion_labels") != [0, 1]:
            raise ValueError("confusion matrices must use the fixed label order [0, 1]")
        matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(2):
            for column in range(2):
                color = "white" if matrix[row, column] > threshold else "black"
                ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color=color)
        ax.set_xticks(np.arange(2), class_names, rotation=15, ha="right")
        ax.set_yticks(np.arange(2), class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(
            f"{MODEL_LABELS[model_name]} (test n={sum(matrix.ravel())}, "
            f"threshold={metrics['threshold']:.2f})"
        )
    fig.suptitle("AI4I Test-set Confusion Matrices", fontsize=15)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _save_pr_curve(report: dict[str, Any], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    reference_y = np.asarray(
        report["models"]["mlp"]["evaluation_data"]["y_true"], dtype=int
    )
    prevalence = float(reference_y.mean())
    for model_name in MODEL_NAMES:
        model_result = report["models"][model_name]
        y_true = np.asarray(model_result["evaluation_data"]["y_true"], dtype=int)
        scores = np.asarray(model_result["evaluation_data"]["positive_scores"], dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, scores, pos_label=1)
        average_precision = float(model_result["metrics"]["pr_auc"])
        threshold_metrics = model_result["metrics"]
        ax.plot(
            recall,
            precision,
            label=f"{MODEL_LABELS[model_name]} (AP={average_precision:.3f})",
        )
        ax.scatter(
            [threshold_metrics["recall"]],
            [threshold_metrics["precision"]],
            marker="x",
            s=65,
        )
    ax.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        label=f"Test-set failure prevalence ({prevalence:.1%})",
    )
    ax.set_title("AI4I Test-set Precision–Recall Curves (positive class: failure)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _save_roc_curve(report: dict[str, Any], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for model_name in MODEL_NAMES:
        model_result = report["models"][model_name]
        y_true = np.asarray(model_result["evaluation_data"]["y_true"], dtype=int)
        scores = np.asarray(model_result["evaluation_data"]["positive_scores"], dtype=float)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores, pos_label=1)
        auc = float(model_result["metrics"]["roc_auc"])
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{MODEL_LABELS[model_name]} (AUC={auc:.3f})",
        )
        matrix = np.asarray(model_result["metrics"]["confusion_matrix"], dtype=int)
        true_negative, false_positive = matrix[0]
        false_negative, true_positive = matrix[1]
        operating_fpr = false_positive / (false_positive + true_negative)
        operating_tpr = true_positive / (true_positive + false_negative)
        ax.scatter([operating_fpr], [operating_tpr], marker="x", s=65)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random baseline")
    ax.set_title("AI4I Test-set ROC Curves (positive class: failure)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _pca_legend_labels(
    train_y: Any,
    test_y: Any,
) -> tuple[dict[str, str], str]:
    """Build class/split legend labels with counts and the full-data prevalence."""
    train = np.asarray(train_y, dtype=int)
    test = np.asarray(test_y, dtype=int)
    legend = {
        "no_failure_train": f"No failure — train (n={int((train == 0).sum())})",
        "failure_train": f"Failure — train (n={int((train == 1).sum())})",
        "no_failure_test": f"No failure — test (n={int((test == 0).sum())})",
        "failure_test": f"Failure — test (n={int((test == 1).sum())})",
    }
    total = len(train) + len(test)
    failures = int((train == 1).sum() + (test == 1).sum())
    prevalence = failures / total if total else 0.0
    prevalence_note = f"Failure prevalence: {prevalence:.1%} ({failures}/{total})"
    return legend, prevalence_note


def _save_pca_3d(prepared: Any, random_state: int, destination: Path) -> None:
    """Fit PCA on training sensor features only; labels are used only for color."""
    train_sensors = prepared.X_train_raw.loc[:, NUMERIC_FEATURE_COLUMNS]
    test_sensors = prepared.X_test_raw.loc[:, NUMERIC_FEATURE_COLUMNS]
    scaler = StandardScaler().fit(train_sensors)
    train_scaled = scaler.transform(train_sensors)
    test_scaled = scaler.transform(test_sensors)
    pca = PCA(n_components=3, random_state=random_state).fit(train_scaled)
    train_components = pca.transform(train_scaled)
    test_components = pca.transform(test_scaled)
    train_y = np.asarray(prepared.y_train, dtype=int)
    test_y = np.asarray(prepared.y_test, dtype=int)
    legend, prevalence_note = _pca_legend_labels(train_y, test_y)
    colors = {0: "#2563eb", 1: "#f97316"}
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for class_value in (0, 1):
        train_mask = train_y == class_value
        test_mask = test_y == class_value
        ax.scatter(
            train_components[train_mask, 0],
            train_components[train_mask, 1],
            train_components[train_mask, 2],
            c=colors[class_value],
            marker="o",
            alpha=0.38,
            s=18,
            label=legend[
                "no_failure_train" if class_value == 0 else "failure_train"
            ],
        )
        ax.scatter(
            test_components[test_mask, 0],
            test_components[test_mask, 1],
            test_components[test_mask, 2],
            c=colors[class_value],
            marker="^",
            alpha=0.9,
            s=38,
            edgecolors="black",
            linewidths=0.3,
            label=legend["no_failure_test" if class_value == 0 else "failure_test"],
        )
    ratios = pca.explained_variance_ratio_
    ax.set_title("AI4I Sensor PCA (fit on training rows; color = Machine failure)")
    ax.set_xlabel(f"PC1 ({ratios[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.1%} variance)")
    ax.set_zlabel(f"PC3 ({ratios[2]:.1%} variance)")
    ax.legend(title="Circle=train, triangle=test", loc="upper right")
    fig.suptitle(prevalence_note, fontsize=10)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _save_mlp_loss_curve(report: dict[str, Any], destination: Path) -> None:
    losses = report["models"]["mlp"]["training"].get("loss_curve", [])
    if not losses:
        raise ValueError("the saved MLP report does not include its training loss curve")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#7c3aed")
    ax.set_title("MLP Training Loss Curve (training loss; not test loss)")
    ax.set_xlabel("Training iteration (epoch)")
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.25)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _save_permutation_importance(report: dict[str, Any], destination: Path) -> None:
    importance = report["models"]["mlp"].get("permutation_importance")
    if not importance or importance.get("scoring") != "average_precision":
        raise ValueError("the saved MLP report does not include AP permutation importance")
    names = list(importance["feature_names"])
    means = np.asarray(importance["importances_mean"], dtype=float)
    deviations = np.asarray(importance["importances_std"], dtype=float)
    if not (len(names) == len(means) == len(deviations)):
        raise ValueError("permutation importance feature and score lengths differ")
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(
        np.asarray(names)[order],
        means[order],
        xerr=deviations[order],
        color="#0f766e",
        alpha=0.8,
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("MLP Permutation Importance on Held-out Test Set")
    ax.set_xlabel("Decrease in average precision after permutation (mean ± SD)")
    ax.set_ylabel("Original model feature (Type categories grouped)")
    ax.text(
        0.99,
        0.02,
        "Type categories permuted together; post-hoc only, not model tuning",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def create_ai4i_visualizations(
    data: pd.DataFrame,
    model_results: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Create six figures from the AI4I CSV and its saved model report.

    Evaluation curves and confusion matrices use the saved held-out predictions.
    PCA is fitted only on training sensor features. Permutation importance is
    computed and stored during model evaluation, then plotted here without
    refitting or tuning a model.
    """
    prepared, metadata = _validate_report(data, model_results)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "pca_3d": destination_dir / "pca_3d.png",
        "confusion_matrices": destination_dir / "confusion_matrices.png",
        "precision_recall_curve": destination_dir / "precision_recall_curve.png",
        "roc_curve": destination_dir / "roc_curve.png",
        "mlp_loss_curve": destination_dir / "mlp_loss_curve.png",
        "mlp_permutation_importance": destination_dir / "mlp_permutation_importance.png",
    }
    _save_pca_3d(prepared, int(metadata["random_state"]), destinations["pca_3d"])
    _save_confusion_matrices(model_results, destinations["confusion_matrices"])
    _save_pr_curve(model_results, destinations["precision_recall_curve"])
    _save_roc_curve(model_results, destinations["roc_curve"])
    _save_mlp_loss_curve(model_results, destinations["mlp_loss_curve"])
    _save_permutation_importance(
        model_results,
        destinations["mlp_permutation_importance"],
    )
    return {name: str(path) for name, path in destinations.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to AI4I CSV")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_MODEL_RESULTS_PATH,
        help="Saved model-comparison JSON produced by ai4i_modeling.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for AI4I visualizations",
    )
    args = parser.parse_args()
    if not args.results.is_file():
        parser.error(
            f"model results were not found at {args.results}; run ai4i_modeling.py first"
        )
    with args.results.open(encoding="utf-8") as stream:
        report = json.load(stream)
    paths = create_ai4i_visualizations(
        load_ai4i_dataset(args.data),
        report,
        args.output_dir,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
