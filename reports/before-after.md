# Lab4 修正前後比較

- 日期：2026-10-23
- 範圍：只比較 Lab4 確認的兩項：Issue [#10 `renderStoredResults`](https://github.com/Lisheepya/26-ai-new-power/issues/10) 與 Issue [#11 `initPage`](https://github.com/Lisheepya/26-ai-new-power/issues/11)。
- 基線：兩項重構前的 `docs/app.js` 與當時的靜態檢查結果；最終數據以本次完整重跑為準。

## 指標比較

| 指標 | 修正前 | 修正後 | 說明 |
| --- | ---: | ---: | --- |
| ESLint 錯誤 | 2 | 0 | 兩項各移除 1 個 `max-lines-per-function` 錯誤。 |
| ESLint 警告 | 0 | 0 | 全程沒有 ESLint warning。 |
| `renderStoredResults` 複雜度 | 1 | 1 | Cyclomatic complexity；門檻 10，前後均未超標。 |
| `initPage` 複雜度 | 2 | 2 | Cyclomatic complexity；門檻 10，前後均未超標。 |
| 複雜度違規數（門檻 10） | 0 | 0 | 這兩項修正處理的是函式長度，不是複雜度超標。 |
| jscpd 重複率 | 0% | 0% | 重複行及重複 token 比率皆維持 0%。 |
| 測試數 | 48 個主測試 + 31 個 subtests | 48 個主測試 + 31 個 subtests | Python 38 個測試（含 31 個 subtests），Node 10 個；最終全部通過。 |

## 兩項修正的 diff 摘要

| Issue | 檔案與 diff 行數 | 函式行數變化 | 摘要 |
| --- | --- | --- | --- |
| #10 `renderStoredResults` | `docs/app.js`，+27 / −11 行 | 103 → 14 行 | 把 Iris 摘要、AI4I 摘要、指標表、報告連結、圖表清單拆成 5 個小函式；原函式只保留清空容器及協調呼叫。 |
| #11 `initPage` | `docs/app.js`，+97 / −82 行 | 94 → 8 行 | 把表單切換、錯誤呈現、兩種預測結果呈現、提交處理與事件綁定抽成 8 個小函式；原函式只負責初始化流程。 |

diff 行數是依 `docs/app.js` 中各項修改區段計算的新增／刪除實體行；函式行數為修正前後函式範圍的實體行數，包含函式宣告與結尾。

## 最難修的項目

Issue #11 的 `initPage` 最難修。它同時負責 DOM 狀態、模型切換、表單提交、推論、結果呈現及錯誤處理；拆分時必須維持事件綁定與兩種表單共用結果區的行為。相較之下，#10 主要是把既有靜態結果渲染區塊按職責切開。

## 最終驗證

- `pnpm run static-check`：通過；ESLint 0 錯誤／0 警告、jscpd 0% 重複率、Knip 0 項。
- `.venv\\Scripts\\python.exe -m pytest -q`：38 passed，31 subtests passed。
- `node --test tests/test_docs_page.cjs tests/test_frontend_inference.cjs tests/test_experiment_conclusions.cjs`：10 passed。
