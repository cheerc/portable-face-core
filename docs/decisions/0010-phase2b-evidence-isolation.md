# 0010. Phase 2B 研究證據與推論隔離

- Status: Accepted
- Date: 2026-09-16
- Relates: 0008（Mac 動態辨識研究）、0009（前處理長寬比不變性）
- Scope: 僅 Phase 2B 研究迴路；不改變任何 Face Core 辨識契約

## Context

Phase 2A 結案時證明了工具正確性與前處理不變性，但沒有回答「真實 session 裡認得對不對」。Phase 2B 要產生那份證據。

量測辨識的研究迴路，有幾種失敗看起來會像成功。三種已經在現行程式碼裡看得到：

- 研究 attempt 記錄建立於相機與模型開啟之後（`src/facecore/research/cli.py`），所以在 device-open 就失敗的嘗試從未進入報表。以此為基礎的任何分母，都偏向那些走得夠遠而被計入的 run。
- `--fixed-seconds` 旗標被接受但沒有實際作用，且 engine 一到終局 controller 就釋放來源。兩條對照臂因此會看到長度不同的窗口。
- 回放從第一張存幀重新推導 session 起點，並以回放時鐘重蓋處理時間，所以一次回放的決策不構成「現場當時如此判定」的證據。

這三項都不是辨識錯誤，卻都會安靜地污染辨識結論。

同一條迴路也處理本專案最敏感的資料。ground truth、切分歸屬與逐幀診斷，正是讓失敗變得可解釋的材料——也正是一旦回流決策路徑，就會製造出它本應量測之結果的材料。

## Decision

1. **ground truth 與切分歸屬永不進入推論。** label、參與者身份與 development／holdout 歸屬由 evaluator 在終局鎖定後寫入，只供分析讀取。scorer、quality rank、採集提示、支持規則與選幀一律不得讀。
2. **attempt 帳本是權威，且先於 session 中會失敗的部分。** 已同意的 Start 被接受時即持久記錄一筆研究 attempt，先於開相機、先於載入模型。open failure、取消、逾時與錯誤都留在分母裡；報表要縮小分母，只能具名列出排除項。
3. **比較需要可被證明的共同窗口。** 採集窗口與推論終局是分開的生命週期事件。固定窗口比較只能由採集 provenance 宣稱，不得從幀數反推。
4. **三種 replay 各自具名，不混用。** 決策重演保留原相對時序並斷言原終局；同幀離線重推論永遠不報成現場決策的重現；反事實只在已批准的 development 實驗下更動因子。
5. **診斷是敏感研究資料。** 逐幀診斷不含 pixels、crop 或 embedding，加密存放於 repo 外的研究 store，沿用既有 record／image 同意與 TTL 分離，並隨 session 鏈一併刪除。彙總報表與 PR body 不帶逐人明細。

## Consequences

Phase 2B 的工程從證據管線開始，而不是從辨識改動開始；前幾個工作包的存在目的是讓錯誤答案現形，不是讓任何答案變好。會讓數字變好、卻未先讓它可量測的工作，排在該管線之後。

本決策之前錄下的既有 bundle 保留其結果，但標示為缺 trace、窗口未證；不以假設值回填。

本決策約束的是研究迴路。它不改變辨識 policy、門檻、gallery 或 learning confirmation 規則，也不授予任何向參與者採集的權限。
