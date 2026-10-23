"""Explore the UCI Iris data, train an MLP classifier, and write report artifacts."""

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "iris.data"
OUTPUT_DIR = ROOT_DIR / "outputs"

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
FEATURE_DESCRIPTIONS = {
    "sepal_length_cm": "萼片長度（sepal length）",
    "sepal_width_cm": "萼片寬度（sepal width）",
    "petal_length_cm": "花瓣長度（petal length）",
    "petal_width_cm": "花瓣寬度（petal width）",
}
TARGET_COLUMN = "species"
CLASS_LABELS = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the headerless UCI CSV and return named measurement and target columns."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Iris data file not found: {dataset_path}")

    data = pd.read_csv(
        dataset_path,
        header=None,
        names=[*FEATURE_COLUMNS, TARGET_COLUMN],
        skipinitialspace=True,
    )
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("string").str.strip()
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def inspect_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize schema, completeness, duplicates, class balance, and feature ranges."""
    class_counts = {
        str(label): int(count)
        for label, count in data[TARGET_COLUMN].value_counts(dropna=False).sort_index().items()
    }
    feature_summary = {}
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        feature_summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "feature_count": len(FEATURE_COLUMNS),
        "column_types": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": int(data.isna().sum().sum()),
        "missing_by_column": {column: int(count) for column, count in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "class_counts": class_counts,
        "feature_summary": feature_summary,
    }


def split_dataset(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Create a reproducible stratified split while preserving feature column names."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def project_pca_3d(features: pd.DataFrame | np.ndarray):
    """Standardize measurements and return the first three PCA coordinates and ratios."""
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=3, svd_solver="full")
    coordinates = pca.fit_transform(standardized)
    return coordinates, pca.explained_variance_ratio_


def _train_and_evaluate(data: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    X_train, X_test, y_train, y_test = split_dataset(data)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.01,
                    max_iter=1500,
                    tol=1e-6,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    # Iris is tiny; limiting BLAS threads prevents oversubscription overhead on
    # machines where the runtime exposes many logical CPUs.
    with threadpool_limits(limits=1):
        model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    per_class = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    mlp = model.named_steps["mlp"]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "class_labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": per_class,
        "loss_curve": [float(loss) for loss in mlp.loss_curve_],
        "iterations": int(mlp.n_iter_),
        "final_training_loss": float(mlp.loss_),
        "model": {
            "type": "MLPClassifier",
            "hidden_layer_sizes": [16, 8],
            "activation": "relu",
            "solver": "adam",
            "max_iter": 1500,
            "learning_rate_init": 0.01,
        },
    }
    return model, metrics


def _save_pca_plot(
    data: pd.DataFrame,
    coordinates: np.ndarray,
    explained_variance: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    palette = dict(zip(CLASS_LABELS, ("#2563eb", "#f59e0b", "#10b981")))
    for label in CLASS_LABELS:
        mask = data[TARGET_COLUMN].to_numpy() == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            coordinates[mask, 2],
            label=label.replace("Iris-", ""),
            color=palette[label],
            s=42,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained_variance[2] * 100:.1f}%)")
    ax.set_title("Iris feature distribution after 3D PCA")
    ax.legend(title="Species")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_training_plot(metrics: dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(metrics["loss_curve"]) + 1), metrics["loss_curve"], color="#2563eb")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("MLP training loss curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_evaluation_plots(metrics: dict[str, Any], output_dir: Path) -> None:
    labels = [label.replace("Iris-", "") for label in metrics["class_labels"]]
    matrix = np.asarray(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = metrics["classification_report"]
    scores = [report[label] for label in metrics["class_labels"]]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, key, color in (
        (-width, "precision", "#2563eb"),
        (0, "recall", "#f59e0b"),
        (width, "f1-score", "#10b981"),
    ):
        ax.bar(x + offset, [score[key] for score in scores], width, label=key, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class classification metrics (test set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "classification_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_report(summary: dict[str, Any], metrics: dict[str, Any], explained: np.ndarray) -> str:
    class_rows = "\n".join(
        f"| {label} | {count} |" for label, count in summary["class_counts"].items()
    )
    feature_rows = "\n".join(
        f"| `{column}` | {FEATURE_DESCRIPTIONS[column]} | {summary['feature_summary'][column]['mean']:.2f} | "
        f"{summary['feature_summary'][column]['min']:.1f}–"
        f"{summary['feature_summary'][column]['max']:.1f} |"
        for column in FEATURE_COLUMNS
    )
    matrix_rows = "\n".join(
        f"| {label.replace('Iris-', '')} | " + " | ".join(str(value) for value in row) + " |"
        for label, row in zip(metrics["class_labels"], metrics["confusion_matrix"])
    )
    report = metrics["classification_report"]
    class_metric_rows = "\n".join(
        f"| {label.replace('Iris-', '')} | {report[label]['precision']:.3f} | "
        f"{report[label]['recall']:.3f} | {report[label]['f1-score']:.3f} | {int(report[label]['support'])} |"
        for label in metrics["class_labels"]
    )
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    cumulative_variance = float(np.sum(explained))
    return f"""# Iris 資料探索與神經網路分類：一項可重現的基準實驗

## 摘要

本研究以 UCI Machine Learning Repository 的 Iris 資料集探索四項花朵測量特徵，使用主成分分析（PCA）呈現三維特徵分布，並以標準化與多層感知器（MLP）建立三分類模型。資料共 {summary['rows']} 筆，分屬三個平衡類別；資料品質檢查發現 {summary['missing_values']} 個缺失值與 {summary['duplicate_rows']} 筆重複列。以固定亂數種子進行分層 80/20 切分後，測試集準確率為 **{metrics['accuracy']:.3f}**（{metrics['test_samples']} 筆）。本結果顯示此小型資料集適合示範探索式視覺化與監督式分類流程，但單次切分的分數不應視為普遍效能保證。

**關鍵詞：** Iris、PCA、多層感知器、分類、模型評估

## 一、研究目的

鳶尾花資料集常用於示範分類方法。本實驗依序檢查資料欄位與品質、觀察不同類別在三個主成分上的分布，並評估類神經網路能否由萼片和花瓣的量測值辨識物種。

## 二、資料與方法

### 2.1 資料集

資料取自 UCI Machine Learning Repository 的 Iris 資料集（Fisher, 1936；CC BY 4.0）。原始檔 `data/iris.data` 為無標題列的 CSV。資料含 {summary['rows']} 筆、四個連續數值特徵及一個類別目標；每個類別各有 50 筆。

| 欄位 | 說明 | 全體平均（公分） | 最小–最大（公分） |
|---|---|---:|---:|
{feature_rows}

目標欄位為 `species`，類別分布如下：

| 分類標籤 | 筆數 |
|---|---:|
{class_rows}

資料完整性檢查確認特徵和目標欄位共 {len(summary['columns'])} 欄，缺失值 {summary['missing_values']} 個、完全重複列 {summary['duplicate_rows']} 筆。欄位型態為：`{', '.join(f'{key}: {value}' for key, value in summary['column_types'].items())}`。

### 2.2 三維 PCA 視覺化

先將四個特徵以標準分數轉換，再以 PCA 投影到前三個主成分。前三個主成分依序解釋 **{explained[0] * 100:.2f}%**、**{explained[1] * 100:.2f}%**、**{explained[2] * 100:.2f}%** 的變異，累積解釋變異為 **{cumulative_variance * 100:.2f}%**。圖中以顏色表示物種；此圖用於探索結構，不作為模型測試資料的評估。

![Iris 三維 PCA 分布](pca_3d.png)

### 2.3 分類器與評估方式

以固定亂數種子 42 執行分層抽樣，保留 80%（{metrics['train_samples']} 筆）作為訓練集、20%（{metrics['test_samples']} 筆）作為測試集。分類流程先使用 `StandardScaler`，再訓練含 16 與 8 個神經元兩個隱藏層的 MLP，採 ReLU、Adam、初始學習率 0.01。標準化器只在訓練資料上擬合，避免測試資料資訊洩漏。以準確率、各類別 precision/recall/F1 和混淆矩陣評估模型。

## 三、實驗結果

模型在 {metrics['iterations']} 次迭代後停止，最後訓練損失為 {metrics['final_training_loss']:.5f}。測試集準確率為 **{metrics['accuracy']:.3f}**，macro-average precision、recall、F1 分數分別為 **{macro['precision']:.3f}**、**{macro['recall']:.3f}**、**{macro['f1-score']:.3f}**；weighted-average F1 為 **{weighted['f1-score']:.3f}**。

| 真實分類 \\ 預測分類 | {metrics['class_labels'][0].replace('Iris-', '')} | {metrics['class_labels'][1].replace('Iris-', '')} | {metrics['class_labels'][2].replace('Iris-', '')} |
|---|---:|---:|---:|
{matrix_rows}

![MLP 訓練損失曲線](training_loss.png)

![測試集混淆矩陣](confusion_matrix.png)

![各類別分類指標](classification_metrics.png)

## 四、討論

類別平衡且沒有缺失值，讓本資料集適合直接進行基準分類。PCA 將四維特徵濃縮為三維，同時保留 {cumulative_variance * 100:.2f}% 的標準化特徵變異，可用來觀察類別群聚與重疊。混淆矩陣及各類別指標應搭配閱讀：若錯誤集中在特定兩類，代表其測量特徵在本次切分中較難區分。MLP 的結果受切分、初始化和超參數影響；本研究使用單一固定切分且未進行系統化調參，因此不能據此比較所有模型或推論部署效能。

## 五、結論

本實驗完成資料探索、三維 PCA、神經網路訓練與測試集評估，並產生訓練歷程、混淆矩陣及分類指標圖。固定種子的結果可由專案程式重現；後續可用分層交叉驗證與不同基準模型，檢查分數對資料切分的敏感度。

## 參考資料

Fisher, R. A. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76. 資料集說明與欄位資訊：https://archive.ics.uci.edu/dataset/53/iris。License: CC BY 4.0.

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html.
"""


def run_analysis(
    data_path: Path | str = DATA_PATH,
    output_dir: Path | str = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the full analysis and write figures, machine-readable metrics, and mini-paper."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data = load_dataset(data_path)
    summary = inspect_dataset(data)
    coordinates, explained_variance = project_pca_3d(data[FEATURE_COLUMNS])
    model, metrics = _train_and_evaluate(data)
    del model

    _save_pca_plot(data, coordinates, explained_variance, output_path / "pca_3d.png")
    _save_training_plot(metrics, output_path / "training_loss.png")
    _save_evaluation_plots(metrics, output_path)
    payload = {
        "dataset_source": "UCI Machine Learning Repository, Iris (DOI: 10.24432/C56C76)",
        "dataset_summary": summary,
        "pca_explained_variance_ratio": [float(value) for value in explained_variance],
        **{key: value for key, value in metrics.items() if key != "loss_curve"},
    }
    (output_path / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "小論文.md").write_text(
        _build_report(summary, metrics, explained_variance), encoding="utf-8"
    )
    return {"summary": summary, "metrics": metrics, "output_dir": output_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the headerless UCI Iris CSV")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated artifacts")
    args = parser.parse_args()
    result = run_analysis(args.data, args.output_dir)
    metrics = result["metrics"]
    print(f"Rows: {result['summary']['rows']}")
    print(f"Missing values: {result['summary']['missing_values']}")
    print(f"Duplicate rows: {result['summary']['duplicate_rows']}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Iterations: {metrics['iterations']}")
    print(f"Artifacts written to: {result['output_dir']}")


if __name__ == "__main__":
    main()
