# `docs/`、`scripts/` 與 `tests/` 靜態檢查報告

- 檢查日期：2026-10-23
- 範圍：`docs/`、`scripts/`、`tests/` 中的 JavaScript（`.js`、`.cjs`、`.mjs`）；不含 HTML、CSS、JSON 與圖片
- 檢查工具：ESLint 10.13.0、eslint-plugin-sonarjs 4.2.2、jscpd 4.3.0、Knip 5.64.1
- 本報告記錄 Lab4 靜態檢查項目的逐項修正與驗證結果；詳見「Lab4 逐項修正紀錄」

## 檢查摘要

| 工具 | 結果 |
| --- | --- |
| ESLint + SonarJS | 0 個錯誤、0 個警告 |
| jscpd | 5 個 JavaScript 檔案、796 行、12,039 tokens；0 個重複片段，行與 token 重複率皆為 0% |
| Knip | 0 個未使用檔案、0 個未使用匯出／相依項目 |
| ESLint `no-unused-vars` | 0 項未使用變數或函式 |

## 1. 警告與錯誤總數

下表按規則分組。修正 #11 後，ESLint 沒有 error 或 warning；其他列出的規則亦未回報項目。

| 規則 | 門檻 | 錯誤 | 警告 | 位置與內容 |
| --- | ---: | ---: | ---: | --- |
| `max-lines-per-function` | 50 行 | 0 | 0 | 無 |
| `complexity` | 10 | 0 | 0 | 無 |
| `max-depth` | 3 | 0 | 0 | 無 |
| `sonarjs/cognitive-complexity` | 15 | 0 | 0 | 無 |
| `no-unused-vars` | ESLint recommended | 0 | 0 | 無未使用變數或函式 |
| `no-undef` | ESLint recommended | 0 | 0 | 無未宣告變數 |
| **合計** |  | **0** | **0** | 兩項 Lab4 函式長度違規均已修正 |

追蹤單：[Issue #10：`renderStoredResults`](https://github.com/Lisheepya/26-ai-new-power/issues/10)、[Issue #11：`initPage`](https://github.com/Lisheepya/26-ai-new-power/issues/11)。

## 2. 重複率與片段位置

jscpd 以 JavaScript 格式掃描 `docs/`、`scripts/` 與 `tests/`，設定最小片段為 5 行及 50 tokens。

| 指標 | 結果 |
| --- | ---: |
| 掃描檔案 | 5 |
| 總行數 | 796 |
| 總 tokens | 12,039 |
| 重複片段 | 0 |
| 重複行／重複率 | 0 行／0% |
| 重複 tokens／重複率 | 0／0% |

本次沒有重複片段，因此沒有重複位置可列。

## 3. 未使用的檔案與函式

- Knip：未使用檔案 0 個，未使用匯出／相依項目 0 個。
- ESLint `no-unused-vars`：未使用變數或函式 0 個。
- 結論：在本次 `docs/`、`scripts/`、`tests/` JavaScript 分析範圍內，沒有靜態工具判定的未使用檔案或函式。

## 4. 前端推論程式規模（Lab4 重構前基線）

以下計數保留自 Lab4 重構前基線，範圍包含 `docs/app.js` 的模型參數、輸入驗證、神經網路計算、兩個預測函式、機率顯示及表單事件流程；不含既有實驗結果的載入與圖表顯示。使用 Espree AST 計算函式範圍，行數以所選程式碼的不同實體行計算，並另列非空行數。

| 範圍 | 函式數 | 實體行數 | 非空行數 |
| --- | ---: | ---: | ---: |
| 前端推論相關程式（含模型參數） | 8 個具名函式；22 個 AST 函式節點（含巢狀函式與箭頭回呼） | 196 | 186 |
| 神經網路計算核心：輸入驗證、前向傳播、啟用函式、Iris／AI4I 預測 | 6 個具名函式 | 70 | 70 |

前端推論的 8 個具名函式為 `requireFiniteNumbers`、`runHiddenLayers`、`softmax`、`sigmoid`、`predictIris`、`predictAi4i`、`renderProbabilityList`、`initPage`。模型參數位於第 2 行；函式實作涵蓋第 167–273 行與第 275–368 行，其中以函式 AST 範圍排除函式間空行。

## 檢查方式

- ESLint：`pnpm run static-check` 中以 ESLint 檢查 `docs/`、`scripts/`、`tests/`。
- jscpd：掃描 `docs/`、`scripts/`、`tests/` 的 `.js`、`.cjs`、`.mjs`，門檻為 5 行／50 tokens。
- Knip：由 `scripts/check-static.cjs` 執行，未回報未使用檔案、匯出或相依項目。

## Lab4 逐項修正紀錄

| 項目 | 修正內容 | ESLint 錯誤／警告變化 | jscpd／Knip 變化 | 測試結果 |
| --- | --- | --- | --- | --- |
| #10 `renderStoredResults` | 拆成 Iris 摘要、AI4I 摘要、模型指標表、報告連結與圖表清單等小函式；原函式只負責清空容器並協調呼叫。未改測試斷言，未使用 `eslint-disable`。 | 2 錯誤、0 警告 → 1 錯誤、0 警告（減少 1 項錯誤） | 重複片段 0 → 0；Knip 發現 0 → 0 | Python 38 項通過（另 31 個 subtests 通過）；Node 10 項通過。全數通過。 |
| #11 `initPage` | 抽出表單切換、錯誤顯示、Iris／AI4I 結果呈現、表單處理與事件綁定函式；`initPage` 保留初始化協調責任。未改測試斷言，未使用 `eslint-disable`。 | 1 錯誤、0 警告 → 0 錯誤、0 警告（減少 1 項錯誤） | 重複片段 0 → 0；Knip 發現 0 → 0 | Python 38 項通過（另 31 個 subtests 通過）；Node 10 項通過。全數通過。 |
