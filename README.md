# Iris 分析與神經網路分類

本專案取得 UCI Iris Dataset，探索四項花朵測量特徵與三種鳶尾花分類，使用 PCA 繪製三維分布，並以標準化加多層感知器（MLP）完成分類實驗。分析程式會產生圖表、JSON 指標和繁體中文小論文。

## 環境與執行

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python iris_analysis.py
python -m unittest discover -s tests -v
```

若 Windows 的 `py` 啟動器不可用，請改用已安裝的 Python 3.10 以上版本建立虛擬環境。

## 專案輸出

- `data/iris.data`：UCI 提供的原始 Iris 資料檔。
- `outputs/pca_3d.png`：標準化特徵的前三個主成分三維散佈圖。
- `outputs/training_loss.png`：MLP 訓練損失曲線。
- `outputs/confusion_matrix.png`：測試集混淆矩陣。
- `outputs/metrics.json`：分類指標及資料探索摘要。
- `outputs/小論文.md`：依實際執行結果生成的小論文。
- `data/ai4i2020.csv`：AI4I 2020 預測性維護資料集原始 CSV。
- `outputs/ai4i_data_validation.json`：AI4I 欄位、型別、缺漏值與重複列檢查報告。
- `outputs/ai4i_eda/`：AI4I 探索摘要與故障標籤、產品類型及感測器圖表。
- `outputs/ai4i_modeling/model_comparison.json`：AI4I Logistic Regression 與 MLP 的模型設定、切分資訊和測試集評估結果。
- `outputs/ai4i/`：AI4I PCA、混淆矩陣、PR/ROC 曲線、MLP 訓練 loss 與 permutation importance 圖表。

## AI4I 2020 資料取得與檢查

從 UCI 官方來源下載資料、驗證 10,000 筆資料與 14 個欄位，並產生檢查報告：

```powershell
python ai4i_data.py --download
```

若已經有 `data/ai4i2020.csv`，可只重新執行本機資料檢查：

```powershell
python ai4i_data.py
```

資料欄位包括識別欄位 `UDI`、`Product ID`，產品類型 `Type`，五項溫度、轉速、扭力與刀具磨耗量測，以及整體故障 `Machine failure` 和故障模式 `TWF`、`HDF`、`PWF`、`OSF`、`RNF`。本資料為合成資料；下載與分享時保留 UCI 出處，依 CC BY 4.0 授權使用。

產生 EDA 摘要及圖表：

```powershell
python ai4i_eda.py
```

前處理 API 會以 `Machine failure` 為目標，只取產品類型和五項感測器特徵，排除識別欄位與故障模式標籤；資料採固定亂數種子的分層切分，編碼器與縮放器只在訓練集擬合：

```python
from ai4i_data import load_ai4i_dataset
from ai4i_preprocessing import prepare_ai4i_data

prepared = prepare_ai4i_data(load_ai4i_dataset())
print(prepared.X_train.shape, prepared.X_test.shape)
```

## AI4I 故障分類模型比較

以同一份固定亂數種子的 80/20 分層切分比較 Logistic Regression 與 MLP。兩模型共用只在訓練資料擬合的前處理；測試資料只用於最後評估，不進行測試集參數或閾值調整。正類為 `Machine failure = 1`，預設機率閾值為 0.5。執行以下指令會將模型設定、切分類別數、評估指標及混淆矩陣寫入 JSON：

```powershell
python ai4i_modeling.py
```

指標包括 accuracy、precision、recall、F1、ROC-AUC 與以 average precision 計算的 PR-AUC。此資料故障樣本僅占 3.39%，因此 accuracy 單獨可能掩蓋漏報；應同時檢視召回率、PR-AUC 與混淆矩陣。以資料集、random state 42、固定閾值 0.5 的本次結果為例，Logistic Regression 的 accuracy 為 96.75%、recall 為 10.29%、F1 為 0.177、PR-AUC 為 0.421；MLP 分別為 97.95%、55.88%、0.650、0.674。這是合成資料上的單次固定切分實驗，不能直接視為實際產線的預測能力估計。

使用已保存的比較 JSON 與 CSV 產生圖表：

```powershell
python ai4i_visualization.py
```

流程會從比較 JSON 讀取測試標籤與預測機率、MLP 訓練 loss 及 permutation importance，不另行訓練或調整模型。PCA 只使用訓練切分的五項感測器特徵擬合，故障標籤只用於著色；Permutation importance 以測試集 average precision 作事後解釋，不用於模型調整。

產生的圖表：

- [感測器 PCA 3D 分布](outputs/ai4i/pca_3d.png)
- [Logistic Regression 與 MLP 混淆矩陣](outputs/ai4i/confusion_matrices.png)
- [Precision–Recall 曲線](outputs/ai4i/precision_recall_curve.png)
- [ROC 曲線](outputs/ai4i/roc_curve.png)
- [MLP 訓練 loss 曲線](outputs/ai4i/mlp_loss_curve.png)
- [MLP permutation importance](outputs/ai4i/mlp_permutation_importance.png)

## 資料來源

- Fisher, R. A. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76. License: CC BY 4.0. UCI 資料集頁面與欄位說明：https://archive.ics.uci.edu/dataset/53/iris。
- UCI Machine Learning Repository. (2020). *AI4I 2020 Predictive Maintenance Dataset* [Dataset]. https://doi.org/10.24432/C5HS5C. License: CC BY 4.0. [資料集頁面與欄位說明](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)。
