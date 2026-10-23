"""Build a Traditional Chinese AI4I paper from saved EDA and model results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_EDA_SUMMARY_PATH = ROOT_DIR / "outputs" / "ai4i_eda" / "summary.json"
DEFAULT_MODEL_RESULTS_PATH = (
    ROOT_DIR / "outputs" / "ai4i_modeling" / "model_comparison.json"
)
DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "AI4I_小論文.md"
FIGURE_DIR = ROOT_DIR / "outputs" / "ai4i"
FIGURES = {
    "pca_3d.png": "AI4I 五項感測器特徵的 PCA 三維投影，依故障標籤與資料分割著色",
    "confusion_matrices.png": "Logistic Regression 與 MLP 的測試集混淆矩陣",
    "precision_recall_curve.png": "兩個模型的測試集 Precision–Recall 曲線",
    "roc_curve.png": "兩個模型的測試集 ROC 曲線",
    "mlp_loss_curve.png": "MLP 訓練損失曲線",
    "mlp_permutation_importance.png": "MLP permutation importance（平均精確率下降）",
}
FIELD_DESCRIPTIONS = {
    "UDI": ("唯一資料列識別碼", "排除"),
    "Product ID": ("產品識別碼，含產品類型與序號", "排除"),
    "Type": ("產品品質類型（L、M、H）", "模型特徵"),
    "Air temperature [K]": ("空氣溫度（K）", "模型特徵"),
    "Process temperature [K]": ("製程溫度（K）", "模型特徵"),
    "Rotational speed [rpm]": ("旋轉速度（rpm）", "模型特徵"),
    "Torque [Nm]": ("扭力（Nm）", "模型特徵"),
    "Tool wear [min]": ("刀具磨耗時間（min）", "模型特徵"),
    "Machine failure": ("整體機器故障標籤（0/1）", "分類目標"),
    "TWF": ("刀具磨耗故障標籤", "排除：避免標籤洩漏"),
    "HDF": ("散熱故障標籤", "排除：避免標籤洩漏"),
    "PWF": ("功率故障標籤", "排除：避免標籤洩漏"),
    "OSF": ("過度應力故障標籤", "排除：避免標籤洩漏"),
    "RNF": ("隨機故障標籤", "排除：避免標籤洩漏"),
}


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _metric(model: dict[str, Any], name: str) -> float:
    try:
        value = float(model["metrics"][name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"model results are missing metric {name!r}") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"metric {name!r} must be between 0 and 1")
    return value


def _confusion_matrix(model: dict[str, Any], model_name: str) -> list[list[int]]:
    try:
        matrix = model["metrics"]["confusion_matrix"]
        labels = model["metrics"]["confusion_labels"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{model_name} results are missing the confusion matrix") from exc
    if labels != [0, 1] or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError(f"{model_name} confusion matrix must use the fixed [0, 1] order")
    return [[int(value) for value in row] for row in matrix]


def _image_markdown(filename: str, alt: str, output_dir: Path) -> str:
    source = FIGURE_DIR / filename
    if not source.is_file():
        raise FileNotFoundError(f"required AI4I report figure is missing: {source}")
    relative = Path(os.path.relpath(source, output_dir)).as_posix()
    return f"![{alt}]({relative})"


def _format_metric(value: float, *, percent: bool = False) -> str:
    return f"{value:.2%}" if percent else f"{value:.3f}"


def render_ai4i_paper(
    eda_summary: dict[str, Any],
    model_results: dict[str, Any],
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_PATH.parent,
    eda_summary_path: Path | str = DEFAULT_EDA_SUMMARY_PATH,
    model_results_path: Path | str = DEFAULT_MODEL_RESULTS_PATH,
) -> str:
    """Render the paper using the saved EDA summary and model report as sources."""
    eda_summary = _require_mapping(eda_summary, "EDA summary")
    model_results = _require_mapping(model_results, "model results")
    metadata = _require_mapping(model_results.get("metadata"), "model metadata")
    models = _require_mapping(model_results.get("models"), "models")
    dataset = _require_mapping(model_results.get("dataset"), "dataset metadata")
    split = _require_mapping(metadata.get("split"), "split metadata")

    if "logistic_regression" not in models or "mlp" not in models:
        raise ValueError("model results must include logistic_regression and mlp")
    if int(eda_summary.get("rows", -1)) != int(dataset.get("rows", -2)):
        raise ValueError("EDA and model reports refer to different row counts")
    eda_fingerprint = eda_summary.get("data_sha256")
    model_fingerprint = dataset.get("data_sha256")
    if not eda_fingerprint or eda_fingerprint != model_fingerprint:
        raise ValueError("EDA and model report data fingerprint does not match")

    logistic = _require_mapping(models["logistic_regression"], "logistic_regression")
    mlp = _require_mapping(models["mlp"], "mlp")
    logistic_metrics = {key: _metric(logistic, key) for key in (
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    )}
    mlp_metrics = {key: _metric(mlp, key) for key in (
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    )}
    logistic_cm = _confusion_matrix(logistic, "Logistic Regression")
    mlp_cm = _confusion_matrix(mlp, "MLP")

    try:
        rows = int(eda_summary["rows"])
        column_count = int(eda_summary["column_count"])
        missing_values = int(eda_summary["missing_values"])
        duplicate_rows = int(eda_summary["duplicate_rows"])
        label_counts = eda_summary["label_counts"]
        failure_count = int(label_counts["Machine failure"]["1"])
        nonfailure_count = int(label_counts["Machine failure"]["0"])
        test_size = float(split["test_size"])
        test_samples = int(split["test_samples"])
        train_samples = int(split["train_samples"])
        random_state = int(metadata["random_state"])
        train_counts = split["train_class_counts"]
        test_counts = split["test_class_counts"]
        failure_modes = {
            label: int(label_counts[label]["1"])
            for label in ("TWF", "HDF", "PWF", "OSF", "RNF")
        }
        correlation = eda_summary["sensor_correlation_matrix"]
        air_process_correlation = float(
            correlation["Air temperature [K]"]["Process temperature [K]"]
        )
        speed_torque_correlation = float(
            correlation["Rotational speed [rpm]"]["Torque [Nm]"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EDA/model reports do not contain the fields required by the paper") from exc

    if rows <= 0 or failure_count + nonfailure_count != rows:
        raise ValueError("Machine failure counts do not match the EDA row count")
    if (
        int(train_counts["1"]) + int(test_counts["1"]) != failure_count
        or int(train_counts["0"]) + int(test_counts["0"]) != nonfailure_count
        or int(train_counts["0"]) + int(train_counts["1"]) != train_samples
        or int(test_counts["0"]) + int(test_counts["1"]) != test_samples
        or train_samples + test_samples != rows
    ):
        raise ValueError("EDA labels and model split class counts do not match")
    if int(metadata.get("positive_class", -1)) != 1:
        raise ValueError("the report expects Machine failure=1 as the positive class")

    prevalence = failure_count / rows
    test_failures = int(test_counts["1"])
    test_nonfailures = int(test_counts["0"])
    majority_baseline = test_nonfailures / test_samples
    model_threshold = float(logistic["metrics"]["threshold"])
    mlp_threshold = float(mlp["metrics"]["threshold"])
    if model_threshold != mlp_threshold:
        raise ValueError("both models must use the same decision threshold")

    figure = lambda filename: _image_markdown(filename, FIGURES[filename], Path(output_dir))
    fields = eda_summary.get("columns", [])
    field_rows = []
    for name in fields:
        description, role = FIELD_DESCRIPTIONS.get(name, ("—", "—"))
        field_rows.append(f"| `{name}` | {description} | {role} |")

    metric_rows = []
    for label, metric_values in (
        ("Logistic Regression", logistic_metrics),
        ("MLP", mlp_metrics),
    ):
        metric_rows.append(
            "| {label} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} | {pr_auc} |".format(
                label=label,
                accuracy=_format_metric(metric_values["accuracy"], percent=True),
                precision=_format_metric(metric_values["precision"], percent=True),
                recall=_format_metric(metric_values["recall"], percent=True),
                f1=_format_metric(metric_values["f1"], percent=True),
                roc_auc=_format_metric(metric_values["roc_auc"]),
                pr_auc=_format_metric(metric_values["pr_auc"]),
            )
        )

    mlp_importance = mlp.get("permutation_importance", {})
    importance_names = mlp_importance.get("feature_names", [])
    importance_means = mlp_importance.get("importances_mean", [])
    importance_stds = mlp_importance.get("importances_std", [])
    if not importance_names or not (len(importance_names) == len(importance_means) == len(importance_stds)):
        raise ValueError("MLP results must include consistent permutation importance arrays")
    importance_rows = [
        f"| `{name}` | {float(mean):.4f} ± {float(std):.4f} |"
        for name, mean, std in sorted(
            zip(importance_names, importance_means, importance_stds),
            key=lambda item: float(item[1]),
            reverse=True,
        )
    ]
    top_importance_name = importance_names[
        max(range(len(importance_means)), key=lambda index: float(importance_means[index]))
    ]

    mlp_training = mlp.get("training", {})
    iterations = mlp_training.get("iterations", 0)
    if isinstance(iterations, list):
        iterations = sum(int(value) for value in iterations)
    convergence = "有" if mlp_training.get("convergence_warning") else "無"
    logistic_params = logistic.get("model_params", {})
    mlp_params = mlp.get("model_params", {})
    hidden_layers = ", ".join(
        str(value) for value in mlp_params.get("hidden_layer_sizes", ())
    )
    early_stopping = "啟用" if mlp_params.get("early_stopping") else "未啟用"
    logistic_penalty = logistic_params.get("penalty", "L2")
    logistic_max_iter = int(logistic_params.get("max_iter", 2000))
    activation = mlp_params.get("activation", "relu")
    solver = mlp_params.get("solver", "adam")
    learning_rate = float(mlp_params.get("learning_rate_init", 0.001))
    alpha = float(mlp_params.get("alpha", 0.0001))
    max_iter = int(mlp_params.get("max_iter", 500))
    validation_fraction = float(mlp_params.get("validation_fraction", 0.1))
    source_url = dataset.get(
        "source",
        "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
    )
    environment = metadata.get("environment", {})
    environment_text = ", ".join(
        f"{name} {version}"
        for name, version in environment.items()
    ) or "版本詳見 `model_comparison.json`"
    modes_text = "、".join(f"{name}={count}" for name, count in failure_modes.items())
    output_dir = Path(output_dir).resolve()
    eda_link = Path(
        os.path.relpath(Path(eda_summary_path).resolve(), output_dir)
    ).as_posix()
    model_link = Path(
        os.path.relpath(Path(model_results_path).resolve(), output_dir)
    ).as_posix()

    content = fr"""# AI4I 2020 預測性維護分類實驗：資料探索與神經網路模型比較

> 本文由 `ai4i_report.py` 根據已保存的 EDA 摘要與模型比較 JSON 產生。主要數值直接取自實驗產物；圖表以相對連結指向 `outputs/ai4i/`。

## 摘要

本研究使用 UCI Machine Learning Repository 的 AI4I 2020 Predictive Maintenance Dataset，探討產品類型與感測器量測能否辨識 `Machine failure`。資料包含 {rows:,} 筆、{column_count} 個欄位，故障樣本 {failure_count:,} 筆（{prevalence:.2%}），類別明顯不平衡。以同一個 {1-test_size:.0%}/{test_size:.0%} 分層訓練／測試切分，比較 Logistic Regression 與多層感知器（MLP），正類為故障、判定閾值為 {model_threshold:.2f}。測試集上 MLP 的 Recall 為 {_format_metric(mlp_metrics['recall'], percent=True)}、ROC-AUC 為 {_format_metric(mlp_metrics['roc_auc'])}、Average Precision（本專案報告欄位稱 PR-AUC）為 {_format_metric(mlp_metrics['pr_auc'])}；Logistic Regression 對應為 {_format_metric(logistic_metrics['recall'], percent=True)}、{_format_metric(logistic_metrics['roc_auc'])}、{_format_metric(logistic_metrics['pr_auc'])}。MLP 表現較佳，但仍有漏判；由於資料是合成資料且僅使用一次隨機切分，本結果不代表真實產線設備的預測能力。

**關鍵詞：** 預測性維護、AI4I 2020、類別不平衡、MLP、故障分類

## 1. 研究背景

預測性維護希望利用設備狀態資訊，在故障發生前辨識風險。本研究以二元分類作為可重現的基準實驗：輸入同一筆觀測中的產品類型與五項感測器特徵，預測該筆觀測是否標記為整體機器故障。由於資料沒有時間戳記或預測提前量，本研究分類的是同筆觀測的故障標籤，**不等於故障發生前的提前預警**。研究問題為：在不使用識別碼或故障模式標籤的條件下，MLP 相較於 Logistic Regression 是否能在固定測試切分上提高故障辨識能力？本實驗比較模型的測試集指標，不建立可直接部署的維修決策系統。

## 2. 資料集與資料探索

資料來自 [UCI AI4I 2020 資料集頁面]({source_url})。UCI 說明此資料為反映產業情境的**合成資料（synthetic data）**，並以 CC BY 4.0 授權發布 [1]。專案保存的 CSV 共 {rows:,} 筆、{column_count} 欄；檢查得到缺漏值 {missing_values} 個、重複列 {duplicate_rows} 筆。分析中的整體標籤 `Machine failure` 有 {nonfailure_count:,} 筆非故障與 {failure_count:,} 筆故障，故障率 {prevalence:.2%}，非故障與故障約為 {nonfailure_count / failure_count:.1f}:1。

| 欄位 | 說明 | 本研究角色 |
|---|---|---|
{chr(10).join(field_rows)}

`TWF`、`HDF`、`PWF`、`OSF`、`RNF` 分別代表刀具磨耗、散熱、功率、過度應力及隨機故障模式；本次資料中正標籤筆數為 {modes_text}。故障模式可同列重疊，這些筆數不可相加當作整體故障筆數。整體分類僅使用 `Machine failure` 作目標，不把模式欄位當輸入。

探索性分析亦顯示，空氣溫度與製程溫度相關係數為 {air_process_correlation:.3f}，轉速與扭力為 {speed_torque_correlation:.3f}。相關性描述此資料內的共同變動，不代表因果關係。

## 3. 研究方法

### 3.1 特徵處理與資料切分

模型輸入為同筆觀測的 `Type` 與五項感測器量測，預測該筆資料的 `Machine failure` 標籤；本研究未定義往前預測的時間窗。排除 `UDI`、`Product ID` 以及五個故障模式標籤，以避免識別資訊與目標相關標籤洩漏。使用固定亂數種子 {random_state} 進行 {1-test_size:.0%}/{test_size:.0%} 分層切分：訓練集 {train_samples:,} 筆（非故障 {train_counts['0']}、故障 {train_counts['1']}），測試集 {test_samples:,} 筆（非故障 {test_nonfailures}、故障 {test_failures}）。`Type` 以 one-hot 編碼，數值特徵以標準化處理，兩者皆只在訓練資料擬合，再轉換測試資料。

### 3.2 分類器與評估

比較 {logistic_penalty} 正則化 Logistic Regression（最多 {logistic_max_iter} 次迭代）與 scikit-learn `MLPClassifier`。MLP 使用隱藏層（{hidden_layers} 個節點）、{activation}、{solver}、初始學習率 {learning_rate:g}、`alpha={alpha:g}`、最多 {max_iter} 次迭代及 early stopping（{early_stopping}；內部分割比例 {validation_fraction:g}）。兩模型使用相同資料切分，固定機率閾值 {model_threshold:.2f}，不重採樣，也不以測試集搜尋超參數或閾值。MLP 本次訓練 {int(iterations)} 次迭代；收斂警告：{convergence}。

報告 accuracy、precision、recall、F1、ROC-AUC 與 PR-AUC。此專案 JSON 中的 PR-AUC 指標以 **Average Precision** 計算，PR 曲線以同一測試集預測分數繪製；這裡不將其解讀為梯形積分近似值。混淆矩陣固定以真實類別為列、預測類別為欄，類別順序皆為 `[0, 1]`。

### 3.3 PCA 與模型解釋圖

PCA 使用五項數值感測器特徵；標準化與 PCA 僅以訓練列擬合，故障標籤只用於圖中著色。PCA 是低維投影，不是分類器的預測結果。Permutation importance 以保留測試集的 Average Precision 下降量作 MLP 事後解釋，`Type` 的 one-hot 欄位一起置換；它不代表因果效果，也不參與模型調整。

## 4. 實驗結果

### 4.1 測試集分類指標

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision（PR-AUC） |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

| 模型 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | {logistic_cm[0][0]} | {logistic_cm[0][1]} | {logistic_cm[1][0]} | {logistic_cm[1][1]} |
| MLP | {mlp_cm[0][0]} | {mlp_cm[0][1]} | {mlp_cm[1][0]} | {mlp_cm[1][1]} |

測試集只要一律預測非故障，Accuracy 即為 {majority_baseline:.2%}；因此高 Accuracy 單獨不足以支持故障辨識能力。固定閾值下，Logistic Regression 找出 {logistic_cm[1][1]} 筆故障、漏掉 {logistic_cm[1][0]} 筆；MLP 找出 {mlp_cm[1][1]} 筆、漏掉 {mlp_cm[1][0]} 筆，並分別產生 {logistic_cm[0][1]} 與 {mlp_cm[0][1]} 筆非故障誤報。

### 4.2 PCA 感測器分布

{figure('pca_3d.png')}

### 4.3 測試集分類曲線與混淆矩陣

{figure('confusion_matrices.png')}

{figure('precision_recall_curve.png')}

{figure('roc_curve.png')}

### 4.4 MLP 訓練與特徵 permutation importance

Loss 圖呈現訓練損失，不是測試損失。

{figure('mlp_loss_curve.png')}

| 原始特徵 | Average Precision 平均下降量（10 次置換，平均 ± 標準差） |
|---|---:|
{chr(10).join(importance_rows)}

{figure('mlp_permutation_importance.png')}

## 5. 討論

在此固定切分與設定下，MLP 的 Recall、F1、ROC-AUC 與 Average Precision 均高於 Logistic Regression，顯示它在本次測試資料上辨識出較多故障樣本。另一方面，MLP 仍漏掉 {mlp_cm[1][0]} / {test_failures} 筆故障，並有 {mlp_cm[0][1]} 筆誤報；是否合適需依漏報與誤報成本及操作閾值評估，不能只據此宣稱已可部署。測試集故障率約 {test_failures / test_samples:.2%}，PR 曲線的基準線因此遠低於高 Accuracy 所暗示的直覺。

本次 MLP permutation importance 中，平均精確率下降量較高的原始特徵為 `{top_importance_name}`。這只表示在此模型、資料切分與評分方式下置換特徵會造成較大分數變化；溫度、轉速、扭力等特徵間的相關性也可能影響重要度排序。PCA 圖則用來檢視感測器資料的低維投影，不足以單獨證明類別在真實製程中可分。

## 6. 研究限制

1. **合成資料限制：** UCI 明確標示資料為合成資料，不能代替真實工廠設備、不同機台或不同故障分布的外部驗證，因此不將本結果外推為產線效能；本分類任務也不是預先設定提前量的故障預警。
2. **單次隨機切分：** 結果來自一個固定種子的分層 holdout，沒有交叉驗證、信賴區間或跨機台測試，對抽樣變化的穩健性尚未量化。
3. **時間泛化未驗證：** 專案 CSV 沒有明確時間戳記，本研究使用隨機列切分；這不能回答依時間先後部署時的預測能力。
4. **閾值與訓練設定固定：** 閾值 {model_threshold:.2f} 及模型參數未經操作成本導向調整。實務上應在獨立驗證資料上選擇閾值並做成本分析。
5. **標籤與解釋限制：** 故障模式標籤可能重疊；Permutation importance 是測試集上的事後模型解釋，不是因果分析。單次 loss curve 只呈現訓練最佳化過程。

## 7. 結論

本研究完成 AI4I 2020 資料檢查、探索、PCA 視覺化，以及 Logistic Regression 與 MLP 的可重現分類比較。在固定測試切分中，MLP 的故障 Recall 為 {_format_metric(mlp_metrics['recall'], percent=True)}，高於 Logistic Regression 的 {_format_metric(logistic_metrics['recall'], percent=True)}，但仍漏掉部分故障。由於樣本為合成資料、故障比例低且僅有一次隨機 holdout，結果適合作為分析流程示範與後續驗證起點，不足以證明模型可直接用於實際設備。

## 8. 重現方式與環境

在專案根目錄使用 Python 3.10 以上版本。若 `py` 啟動器不可用，可改用 `python -m venv .venv`，或指定已安裝的 Python 3.10 以上執行檔建立環境。建立虛擬環境並安裝 `requirements.txt` 後，依序取得／驗證資料、重做 EDA、訓練模型、產生圖表，再生成本報告：

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe ai4i_data.py --download
.venv\Scripts\python.exe ai4i_eda.py
.venv\Scripts\python.exe ai4i_modeling.py --test-size 0.2 --random-state 42
.venv\Scripts\python.exe ai4i_visualization.py
.venv\Scripts\python.exe ai4i_report.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

若已保存 `data/ai4i2020.csv`，可略過下載並執行 `.venv\Scripts\python.exe ai4i_data.py` 驗證本機資料。報告引用 [EDA 摘要]({eda_link}) 與[模型比較 JSON]({model_link})，並核對兩份報告的資料指紋（SHA-256：`{eda_fingerprint}`）；圖表位於 `ai4i/`。本次實驗記錄的環境為 {environment_text}（Python {environment.get('python', '版本記錄於模型 JSON')}）。固定亂數種子可重現資料切分；若套件版本或平台不同，模型數值可能有細微差異，實際版本請以模型 JSON 的環境欄位為準。

## 參考資料

[1] UCI Machine Learning Repository. (2020). *AI4I 2020 Predictive Maintenance Dataset* [資料集]. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). CC BY 4.0. [資料集與變數說明]({source_url})。
"""
    return content


def generate_ai4i_paper(
    *,
    eda_summary_path: Path | str = DEFAULT_EDA_SUMMARY_PATH,
    model_results_path: Path | str = DEFAULT_MODEL_RESULTS_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Read saved experiment artifacts and write the AI4I paper as UTF-8 Markdown."""
    eda_path = Path(eda_summary_path)
    results_path = Path(model_results_path)
    destination = Path(output_path)
    try:
        eda_summary = json.loads(eda_path.read_text(encoding="utf-8"))
        model_results = json.loads(results_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"could not read saved experiment artifact: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment JSON: {exc}") from exc

    content = render_ai4i_paper(
        eda_summary,
        model_results,
        output_dir=destination.parent,
        eda_summary_path=eda_path,
        model_results_path=results_path,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-summary", type=Path, default=DEFAULT_EDA_SUMMARY_PATH)
    parser.add_argument("--model-results", type=Path, default=DEFAULT_MODEL_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    destination = generate_ai4i_paper(
        eda_summary_path=args.eda_summary,
        model_results_path=args.model_results,
        output_path=args.output,
    )
    print(f"AI4I paper: {destination}")


if __name__ == "__main__":
    main()
