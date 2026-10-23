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

## 資料來源

Fisher, R. A. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76. License: CC BY 4.0. UCI 資料集頁面與欄位說明：https://archive.ics.uci.edu/dataset/53/iris。
