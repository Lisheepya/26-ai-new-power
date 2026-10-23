"""Exploratory analysis and visual reports for the UCI AI4I 2020 dataset."""

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
from matplotlib.patches import Patch

from ai4i_data import DATA_PATH, DATASET_PAGE_URL, inspect_ai4i_dataset, load_ai4i_dataset


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "ai4i_eda"
SUMMARY_FILENAME = "summary.json"
LABEL_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
SENSOR_UNITS = {
    "Air temperature [K]": "K",
    "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "Nm",
    "Tool wear [min]": "min",
}


def _as_json_number(value: Any) -> int | float | None:
    """Convert pandas/numpy numeric values to strict JSON-compatible scalars."""
    if pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _describe_columns(data: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column in columns:
        description = data[column].describe()
        result[column] = {
            str(statistic): _as_json_number(value)
            for statistic, value in description.items()
        }
    return result


def summarize_ai4i_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize the schema, data quality, feature statistics, and failure labels."""
    # Keep the canonical AI4I schema and type checks, but report rather than reject
    # missing or duplicated rows so EDA can make those quality issues visible.
    inspect_ai4i_dataset(data, expected_rows=None)

    label_counts: dict[str, dict[str, int]] = {}
    for column in LABEL_COLUMNS:
        counts = data[column].value_counts(dropna=True).to_dict()
        normalized = {str(int(value)): int(count) for value, count in counts.items()}
        normalized.setdefault("0", 0)
        normalized.setdefault("1", 0)
        label_counts[column] = {key: normalized[key] for key in sorted(normalized)}
    label_prevalence = {}
    for column, counts in label_counts.items():
        total = sum(counts.values())
        positive_count = counts.get("1", 0)
        prevalence = positive_count / total if total else 0.0
        label_prevalence[column] = {
            "positive_count": positive_count,
            "labeled_count": total,
            "positive_prevalence": prevalence,
            "positive_prevalence_percent": prevalence * 100,
        }

    type_failure_rates: dict[str, dict[str, int | float]] = {}
    failure_by_type = data.groupby("Type", dropna=False, sort=True)["Machine failure"]
    for type_value, failures in failure_by_type:
        key = "<missing>" if pd.isna(type_value) else str(type_value)
        total = int(failures.count())
        failure_count = int(failures.fillna(0).sum())
        type_failure_rates[key] = {
            "total": total,
            "failures": failure_count,
            "failure_rate": failure_count / total if total else 0.0,
        }

    sensor_correlation = data[SENSOR_COLUMNS].corr()
    sensor_correlation_matrix = {
        str(row): {
            str(column): _as_json_number(value)
            for column, value in sensor_correlation.loc[row].items()
        }
        for row in sensor_correlation.index
    }
    sensor_distributions_by_type: dict[str, dict[str, dict[str, Any]]] = {}
    for type_value, group in data.groupby("Type", dropna=False, sort=True):
        key = "<missing>" if pd.isna(type_value) else str(type_value)
        sensor_distributions_by_type[key] = _describe_columns(group, SENSOR_COLUMNS)

    failure_labels = data["Machine failure"]
    positive_count = int(failure_labels.eq(1).sum())
    negative_count = int(failure_labels.eq(0).sum())
    labeled_count = positive_count + negative_count
    positive_prevalence = positive_count / labeled_count if labeled_count else 0.0
    negative_to_positive_ratio = (
        negative_count / positive_count if positive_count else None
    )
    is_imbalanced = (
        negative_to_positive_ratio is not None and negative_to_positive_ratio >= 2
    )
    balance = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "labeled_count": labeled_count,
        "positive_prevalence": positive_prevalence,
        "positive_prevalence_percent": positive_prevalence * 100,
        "negative_to_positive_ratio": negative_to_positive_ratio,
        "is_imbalanced": is_imbalanced,
    }
    ratio_note = (
        f"{negative_to_positive_ratio:.1f}:1 non-failure to failure examples"
        if negative_to_positive_ratio is not None
        else "no positive machine-failure examples are present"
    )
    balance_note = "class imbalance is present" if is_imbalanced else "classes are not strongly imbalanced"
    prevalence_note = (
        f"Machine failure prevalence is {positive_prevalence * 100:.2f}% "
        f"({positive_count}/{labeled_count}); {balance_note}, with {ratio_note}."
        if labeled_count
        else "Machine failure prevalence cannot be computed because no target labels are available."
    )

    numeric_columns = [
        column for column in data.select_dtypes(include=[np.number]).columns
    ]
    missing_by_column = {
        str(column): int(value) for column, value in data.isna().sum().items()
    }
    return {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "source": DATASET_PAGE_URL,
        "rows": int(len(data)),
        "column_count": int(len(data.columns)),
        "columns": [str(column) for column in data.columns],
        "column_types": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": int(sum(missing_by_column.values())),
        "missing_by_column": missing_by_column,
        "duplicate_rows": int(data.duplicated().sum()),
        "descriptive_statistics": _describe_columns(data, numeric_columns),
        "sensor_distributions": _describe_columns(data, SENSOR_COLUMNS),
        "sensor_distributions_by_type": sensor_distributions_by_type,
        "sensor_correlation_matrix": sensor_correlation_matrix,
        "label_counts": label_counts,
        "label_prevalence": label_prevalence,
        "type_failure_rates": type_failure_rates,
        "machine_failure_balance": balance,
        "summary_notes": [
            "AI4I 2020 is a synthetic (simulated) predictive-maintenance dataset, not direct measurements from an actual production line.",
            prevalence_note,
            "Failure-mode labels can overlap on the same row; do not sum mode counts as if they were mutually exclusive machine failures.",
        ],
    }


def _plot_label_counts(summary: dict[str, Any], destination: Path) -> None:
    prevalence_summary = summary["label_prevalence"]
    labels = list(prevalence_summary)
    positives = [prevalence_summary[label]["positive_count"] for label in labels]
    prevalences = [
        float(prevalence_summary[label]["positive_prevalence_percent"])
        for label in labels
    ]
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars = ax.bar(positions, prevalences, color="#f97316", label="Positive label prevalence")
    ax.set_title("AI4I Positive Failure Label Prevalence")
    ax.set_xlabel("Machine failure / failure mode (mode labels may overlap)")
    ax.set_ylabel("Positive records among labeled rows (%)")
    ax.set_xticks(positions, labels)
    ax.legend(title="Label value: 1 — failure")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0, top=max(1.0, max(prevalences, default=0) * 1.3))
    for bar, count, prevalence in zip(bars, positives, prevalences):
        ax.annotate(
            f"{count} ({prevalence:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4 if prevalence else 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _plot_type_failure_rates(summary: dict[str, Any], destination: Path) -> None:
    rates = summary["type_failure_rates"]
    labels = list(rates)
    percentages = [100 * float(rates[label]["failure_rate"]) for label in labels]

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    bars = ax.bar(labels, percentages, color="#14b8a6", label="Machine failure rate")
    ax.set_title("Machine Failure Rate by Product Type")
    ax.set_xlabel("Product type")
    ax.set_ylabel("Machine failure rate (%)")
    ax.set_ylim(bottom=0, top=max(5, max(percentages, default=0) * 1.18))
    ax.legend(title="Measure")
    ax.grid(axis="y", alpha=0.25)
    for bar, rate in zip(bars, percentages):
        ax.annotate(
            f"{rate:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _plot_sensor_distributions_by_type(data: pd.DataFrame, destination: Path) -> None:
    type_groups = list(data.groupby("Type", dropna=False, sort=True))
    type_labels = ["<missing>" if pd.isna(label) else str(label) for label, _ in type_groups]
    colors = plt.get_cmap("tab10")
    type_colors = {label: colors(index % 10) for index, label in enumerate(type_labels)}
    fig, axes = plt.subplots(
        nrows=len(SENSOR_COLUMNS),
        ncols=1,
        figsize=(9, 13),
        constrained_layout=True,
    )
    for ax, column in zip(axes, SENSOR_COLUMNS):
        grouped_values = [
            group[column].dropna().to_numpy()
            for _, group in type_groups
        ]
        boxplot = ax.boxplot(
            grouped_values,
            patch_artist=True,
            widths=0.55,
        )
        ax.set_xticks(np.arange(1, len(type_labels) + 1), type_labels)
        for patch, label in zip(boxplot["boxes"], type_labels):
            patch.set_facecolor(type_colors[label])
            patch.set_alpha(0.65)
        ax.set_title(column)
        ax.set_ylabel(SENSOR_UNITS[column])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(
        handles=[
            Patch(facecolor=type_colors[label], alpha=0.65, label=label)
            for label in type_labels
        ],
        title="Product type (Type)",
        loc="upper right",
    )
    axes[-1].set_xlabel("Product type (Type)")
    fig.suptitle("Sensor Distributions by Product Type", fontsize=15)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _plot_sensor_correlation(data: pd.DataFrame, destination: Path) -> None:
    correlation = data[SENSOR_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("AI4I Numeric Sensor Correlation Matrix")
    ax.set_xticks(np.arange(len(SENSOR_COLUMNS)), SENSOR_COLUMNS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(SENSOR_COLUMNS)), SENSOR_COLUMNS)
    for row in range(len(SENSOR_COLUMNS)):
        for column in range(len(SENSOR_COLUMNS)):
            value = correlation.iloc[row, column]
            label = f"{value:.2f}" if pd.notna(value) else "n/a"
            ax.text(column, row, label, ha="center", va="center", color="black", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation (unitless)")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def create_ai4i_eda_artifacts(
    data: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
) -> dict[str, Any]:
    """Create the EDA summary JSON and labeled charts in ``output_dir``."""
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_ai4i_dataset(data)
    summary_path = destination_dir / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    chart_paths = {
        "label_counts": destination_dir / "label_counts.png",
        "type_failure_rates": destination_dir / "type_failure_rates.png",
        "sensor_distributions_by_type": destination_dir / "sensor_distributions_by_type.png",
        "sensor_correlation_heatmap": destination_dir / "sensor_correlation_heatmap.png",
    }
    _plot_label_counts(summary, chart_paths["label_counts"])
    _plot_type_failure_rates(summary, chart_paths["type_failure_rates"])
    _plot_sensor_distributions_by_type(data, chart_paths["sensor_distributions_by_type"])
    _plot_sensor_correlation(data, chart_paths["sensor_correlation_heatmap"])
    return {
        "summary": str(summary_path),
        "charts": {name: str(path) for name, path in chart_paths.items()},
        "report": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to AI4I CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where summary JSON and charts are saved",
    )
    args = parser.parse_args()
    data = load_ai4i_dataset(args.data)
    artifacts = create_ai4i_eda_artifacts(data, args.output_dir)
    print(json.dumps(artifacts["report"], ensure_ascii=False, indent=2))
    print(f"EDA summary: {artifacts['summary']}")
    for name, path in artifacts["charts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
