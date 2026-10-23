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

## 資料來源

- Fisher, R. A. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76. License: CC BY 4.0. UCI 資料集頁面與欄位說明：https://archive.ics.uci.edu/dataset/53/iris。
- UCI Machine Learning Repository. (2020). *AI4I 2020 Predictive Maintenance Dataset* [Dataset]. https://doi.org/10.24432/C5HS5C. License: CC BY 4.0. [資料集頁面與欄位說明](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)。
