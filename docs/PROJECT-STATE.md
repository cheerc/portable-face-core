# Project State

更新：2026-09-15。證據基準：PR #56 merge `fab5c43f850d1fc60d9644b86369ce863c99570a`。啟動 session 時仍須查 live main／board；本檔不是 daemon 工作快照。

## 現在在哪裡

- Phase-1A 靜態 pipeline／CLI／Layer-A conformance 已交付。
- Phase-1B plan 的 PR-C–G（Task 1–11）及後續接線／治理修正已 merge；不是「尚未取得開工 go」。P0 已於 decision `d-20260912041106780391-34` 放行，實作全波段授權為 `d-20260912090115587658-42`。
- 真圖 replay 已在 M3／PR #38 接入報表並重跑；不是「只有 synthetic／等待權重」。但真實 candidate creation/promotion 成效及長期 drift 尚未證明，不能宣稱治理已獲全面真實情境驗證。
- SFace Pair 1 維持 provisional，**selection gate OPEN**。M1–M6 工作交付不等於選型條件全部滿足，更不等於安全認證或正式部署核准。
- **Phase 2A 實作鏈 T1–T8 已全數 merge**（PR #44–#51），並補完計畫外三項缺口：真機相機接線 PR #52（T7 的 `--device` 非 fake 分支原為 exit 2 死路，T1–T8 只驗 fake pump）、device 透傳＋live 時鐘修復 PR #53、加密存幀接線＋逐幀 ID ledger＋replay 真 gallery 重建 PR #54。[正式研究設計](specs/2026-09-14-mac-live-identification-research-design.md)／[ADR 0008](decisions/0008-mac-live-identification-research.md)／[實作計畫](plans/2026-09-14-mac-live-identification-implementation-plan.md) 為研究邊界。
- **真人 smoke 已執行（operator 本人，單一參與者）**：六次 session 下來鏈路跑通——相機→取幀→真實 YuNet+SFace 評分→session engine→AEAD 加密鏈→回放→刪除。有效輪（blind-005/006）採到真人幀並產生 per-frame ledger，**但頂候選皆非本人、全部落 review 帶、margin 0.017–0.041，operator 現場判定辨識結果錯誤**；原因已查明為前處理長寬比形變 bug（[ADR 0009](decisions/0009-preprocessing-aspect-invariance.md)，decision `d-20260915071553820286-1` operator go 修復中）。所有 session 已依同意設計全數刪除（零殘留），故目前**沒有任何封存 session 語料**。
- 據此可宣稱：**Phase 2A 工具鏈就緒且已在真機上通過一次端到端執行（執行成功、辨識失敗，失敗原因為已定位的前處理 bug，非模型／門檻／gallery 問題）**。不可宣稱：校準完成、辨識可用、誤認率為 0、準備部署。
- **Phase 2A／2B 界線（2026-09-15 修訂）：** 2A＝工具正確性＋前處理不變性＋團隊可自足的 §9 項目，**不設準確率目標**；2B＝研究有效性（§5 對照臂、§8 分母、holdout 切分、未註冊參與者 session）。未註冊參與者 session 移至 2B，不構成 2A 收尾條件。引導 UI 定案為 **pyside6** 真窗＋方形對齊框（Cocoa 關閉）；方形框為採集側措施，不替代 §6 不變量。

## 功能與 evidence 的界線

已有：decode/detect/align/embed/compare、單幀 policy、加密 SQLite／KeyProvider、identity lifecycle、確認候選與 corroboration/promotion/utility/eviction、rollback/delete/export/import、generation migration、drift、replay、capacity 工具。**Phase 2A 新增**：相機 adapter（真機 device 可用）、即時 session 聚合器、研究 session recorder（同意／加密／TTL／可恢復刪除）、加密存幀與逐幀 ID ledger、研究 CLI（live/replay/delete）。可用 CLI 語法以 `facecore.sh --help` 與目前 parser 為準；母規格的示意 `identify --confirm-learning` 不等於所有使用情境已有端到端產品 UI。

尚未有：**引導 UI（pyside6 真窗未實作，T7 僅交付 headless bindings；pyside6／Cocoa 最終抉擇仍 OPEN）**；既有 replay harness 不替代它。既有人工照片驗收及 synthetic 功能測試，不保證換造型、跨日學習、500 人辨識或相機防翻拍效果。**真機證據目前僅一位參與者、六次 session，且結果為辨識錯誤**——單一 operator 的一輪執行不構成研究有效性證據。

## 現有選型證據（分母與限制不得省略）

| 工作 | 交付來源 | 可支持的結論／限制 |
| --- | --- | --- |
| M4 add 接線 | PR #30；re-enroll PR #32 | 單張真 pipeline 寫入／re-enroll 接線已驗，不是動態掃臉 |
| M6 premise | `d-20260913084623309226-0` | 既有權重與檔名序 B 獲授權；非 EXIF 證實的跨日時間線 |
| M2 Pair 2 | PR #35 | manifest/generation/同協議 bakeoff 已交付；不能由此推論新 30 non-target 的 Pair2 結果已完成 |
| M1 real FA | PR #36 | Pair1、detector 0.8、23 gallery、30 non-target；match 0.45/margin 0.10 時觀察 FA 4/30 |
| M5 二維表 | PR #37 | 7×4 cells；margin 0.15、match 0.30–0.55 時 target 3/13、FA 0/30；match 0.60 時 target 2/13。僅此資料集觀察，不是新產品預設 |
| M3 real replay | PR #38 | 13 個 person-23 探針；baseline matched/review/unknown=2/10/1，adaptive=0/13/0；creation/corroboration/promotion=0，rejections=13 |
| 2A 真機 smoke | 本機執行，無 repo artifact | 一位已註冊參與者、六次 session；鏈路端到端通、per-frame ledger 產出；**頂候選非本人、全落 review、margin ≤0.041**。session 已全刪，不可重算；不支持任何準確率宣稱 |

M3 的 6 個 correct-supervision 事件最高 score 約 0.6785，均低於 candidate update 0.88；其餘 7 個為 not_me。這證明本串流沒有建立 candidate，**不證明學習增益或長期不污染**。M1 的 30 non-target 未包含在這 13 事件中；不能把 4/30 FA 說成那 7 個 not_me 的子集。Replay 的 retirements/rollbacks 固定 close-out 段須與真實輸入觸發分開，不計為現場生命週期成功證據。

**Contract v1 標註（只標註不重詮釋，2026-09-15）：** 上表 M1／M5／M3 皆為 `ALIGN_CONTRACT_VERSION=1` 下的量測——gallery 側方形（形變 1.0）、探針側 1440x1920（形變 1.333）。是否需重詮釋或重測為另行決策，**不在本次授權內**，任何人不得據此觸碰既有選型報表。

**治理缺口曾存在與消解（2026-09-15）：** `ALIGN_CONTRACT_VERSION` 升版在 PR #58 前無治理層消費者——升版不建新 generation、不重嵌、不轉 `re_enrollment_required`（手動重建前例：PR #58 1→2，research gallery digest `e98519ae`→`b3448c31`）。消解：`governance/contract_guard.py` 綁定版本→generation，runtime manifest 派生，升版即 `MIGRATION_REQUIRED`；runtime generation 字串 `sface-112-rgb`→`sface-112-rgb+align2`（本 PR 起）。

Baseline/adaptive 有不同規則；2/13 對 0/13 不是純粹同 operating point 的模型 A/B 勝負。檔名序只為授權的 replay 排序，不代表時間或年齡趨勢。零觀察誤認不代表零真實風險；小 gallery 不外推 500 人。**靜態 M1/M5 成績不推論動態 session 成績，反向亦然；2A smoke 的辨識錯誤不等於靜態 pipeline 迴歸，須由 spike 分離成因後才能歸因。**

原 no-weights [報告骨架](2026-09-12-model-selection-report-skeleton.md) 是歷史證據，不是最新 pending 清單。真實照片、權重、embeddings、DB、可識別逐筆 logs 均留 repo 外；跨 session 應確認外部 artifact 存活與版本，不把 `/tmp` 當永久證據庫。

## 階段與授權，避免重新走錯 gate

1. **1B P0 開工 gate：已完成。** ADR 0006 是 accepted 的分階段架構決策，不是現在等著第一次批准 1B 的待辦。
2. **1B 工程／研究 closeout：** 已交付功能與 M3 真圖負向／閾值 evidence；保留未測的真實 promotion、long-horizon drift 和現場失敗恢復限制。不能僅因 M3 跑完便自動宣布全部 spec acceptance 達成。
3. **選型／operating point：OPEN。** 未決部署準確性不禁止經獨立設計的受控 Mac 研究；研究不降低原有模型 integrity/license/provenance gate。
4. **Phase 2A：實作鏈完成，研究本體未開始。** 實作計畫 §7 的三種完成互不替代——（a）plan 書面完成：已達成；（b）產品實作完成＝T8 工程＋真機證據：工程已交付，真機證據僅單一參與者一輪且結果為錯誤，研究設計 §9 驗收矩陣仍有未滿項（見下）；（c）研究有效性＝封存 session 比較後另行報告：**尚未開始，且目前語料為零**。錄製需個別參與者同意；文件批准與工具就緒都不構成同意，也不構成研究結論。
5. **Android/iOS／認證／產品整合：另行決策。** 代表性 gallery 重校準在部署／目標容量宣稱前完成；跨 runtime 與真人 carrier 在 mobile 評估前完成。相機引導、多幀穩定不等於 liveness。

## 後續範圍

研究第一版固定 one-shot gallery，以 session 為評估單位；保存採明確同意、加密、TTL／刪除，詳細契約只在研究設計中維護。未註冊者、本人認錯身份、timeout 和品質失敗都要計數；先蒐集小批 session，再封存 holdout 比較，無須先手工整理大圖庫。

**研究設計 §9 驗收矩陣未滿項（Phase 2A 收尾必要條件）：** 引導 UI 缺席，故「測試程式輸出與 UI 對上、保存開關明顯」無法驗；已同意的未註冊參與者 session 尚未執行（仍 blocked，需另一位參與者與其個別同意）；真機的 permission denied／disconnect／worker failure／UI close 有限清理僅在 fake device 上驗過。

**研究本體未開始項：** §5 的兩條對照臂（品質最佳單幀基線 vs 時間一致性候選策略）從未在真實 session 上比較；§8 的 session 級分母（attempted／labeled／eligible／successful、time-to-decision、品質拒絕）尚無真實數字；development 與 sealed holdout 的切分未建立。

後續獨立研究：受控離線學習 replay（包含 unknown 與錯認事件）、更代表性圖庫、模型對比、現場效能與處理失敗。待驗項：R4 缺檔 seq/rank 對位、各臂 refused 計數／雜檔處理、ORT teardown crash（不可類比 timing flake）、continuity 位移界線初值（0.50 仍為初值，野外未驗）、ORT 端到端 5fps 餘量、pyside6／Cocoa 最終抉擇、**取幀解析度為 720x1280 直式的成因與對辨識的影響**、**2A smoke 辨識錯誤的成因分離**、**研究 CLI 缺 `__main__` 接線／capacity growth-ratio 滿載 flake／manifest 與 engine 的 frame 計數口徑分離**。這些是已知限制／後續範圍，不隱含派工授權。

## Next Session

1. 讀本檔、母規格、ADR 0008、Mac 研究設計；查 git／task／inbox 活源。
2. Phase 2A 實作鏈狀態：T1–T8（PR #44–#51）＋真機接線 #52＋device/時鐘修復 #53＋存幀與逐幀 ledger #54 全數 merge。真人 smoke 已執行六輪並全數刪除；工具就緒、辨識未驗證。
3. 已立未派任務：解析度偏差 spike、辨識全錯 spike（兩者為 ANALYSIS-only，可能同因）、小缺口三件 patch 批次。研究本體（§5 對照臂、§8 報告分母）與引導 UI 尚未立項。
4. Plan 規範：相機錄製需個別參與者同意；文件批准不構成同意。正式 GUI 入口仍待 pyside6／Cocoa 抉擇。
5. 不重開已完成的 P0；不把歷史骨架的缺權重、未跑真圖或本機舊 main 當現況；不重新派 M1/M2/M3/M5 與 T1–T8 已完成任務。
