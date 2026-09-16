# Project State

更新：2026-09-16。證據基準：main `dab9220c7e1241afddb36679e2b1642e84c73adc`（PR #69）。Phase 2A 結案（decision `d-20260916002300248295-1`）；Phase 2B 文件與研究工具工程已授權開工（decision `d-20260916012440460338-3`），研究能力尚未實作、尚無辨識有效性證據。啟動 session 時仍須查 live main／board；本檔不是 daemon 工作快照。

## 現在在哪裡

- Phase-1A 靜態 pipeline／CLI／Layer-A conformance 已交付。
- Phase-1B plan 的 PR-C–G（Task 1–11）及後續接線／治理修正已 merge；不是「尚未取得開工 go」。P0 已於 decision `d-20260912041106780391-34` 放行，實作全波段授權為 `d-20260912090115587658-42`。
- 真圖 replay 已在 M3／PR #38 接入報表並重跑；不是「只有 synthetic／等待權重」。但真實 candidate creation/promotion 成效及長期 drift 尚未證明，不能宣稱治理已獲全面真實情境驗證。
- SFace Pair 1 維持 provisional，**selection gate OPEN**。M1–M6 工作交付不等於選型條件全部滿足，更不等於安全認證或正式部署核准。
- **Phase 2A 實作鏈 T1–T8 已全數 merge**（PR #44–#51），並補完計畫外三項缺口：真機相機接線 PR #52、device 透傳＋live 時鐘修復 PR #53、加密存幀接線＋逐幀 ID ledger＋replay 真 gallery 重建 PR #54。[正式研究設計](specs/2026-09-14-mac-live-identification-research-design.md)／[ADR 0008](decisions/0008-mac-live-identification-research.md)／[實作計畫](plans/2026-09-14-mac-live-identification-implementation-plan.md) 為研究邊界。
- **Phase 2A 結案（2026-09-16，七項）：** T1–T8 實作鏈；前處理不變性修復 contract v3（letterbox PR #58、5 點對齊 PR-2 PR #60、spec 補正 PR #57）；治理機制 PR #59＋helper 清理 PR #62；真機故障路徑 worker 競態修復 PR #66（exact-HEAD＋landed HEAD 雙真機驗 PASS）；修復後 smoke live-v3-001／v3-002 鏈路通（不以認對驗收）；檢查腳本化 PR #67；本機列舉 PR #68（在場確認 PASS）。
- 真人 session 全部依同意設計刪除（零殘留），目前**沒有任何封存 session 語料**。
- 據此可宣稱：**Phase 2A 工具鏈就緒（含前處理正確性與真機端到端執行）**。不可宣稱：校準完成、辨識可用、誤認率為 0、準備部署。
- **Phase 2A／2B 界線：** 2A＝工具正確性＋前處理不變性＋團隊可自足的 §9 項目，**不設準確率目標**；2B＝研究有效性（§5 對照臂、§8 分母、holdout 切分、未註冊參與者 session）。**2B 的文件（G0）與研究工具工程（G1）已授權；採集（G3）、改善／調參（G4）、holdout 解封（G5）仍未開。** 契約見 [Phase 2B 研究規格](specs/2026-09-16-phase2b-mac-recognition-research.md)。引導 UI 定案為 **pyside6** 真窗＋方形對齊框（Cocoa 關閉）；方形框為採集側措施，不替代 §6 不變量。

## 功能與 evidence 的界線

已有：decode/detect/align/embed/compare、單幀 policy、加密 SQLite／KeyProvider、identity lifecycle、確認候選與 corroboration/promotion/utility/eviction、rollback/delete/export/import、generation migration、drift、replay、capacity 工具。**Phase 2A 新增**：相機 adapter（真機 device 可用＋`local` 列舉解析）、即時 session 聚合器、研究 session recorder（同意／加密／TTL／可恢復刪除）、加密存幀與逐幀 ID ledger、研究 CLI（live/replay/delete）。可用 CLI 語法以 `facecore.sh --help` 與目前 parser 為準；母規格的示意 `identify --confirm-learning` 不等於所有使用情境已有端到端產品 UI。

尚未有：**引導 UI pyside6 真窗未實作（T7 僅 headless bindings）**；既有 replay harness 不替代它。既有人工照片驗收及 synthetic 功能測試，不保證換造型、跨日學習、500 人辨識或相機防翻拍效果。**真機辨識正確性未驗證**——修復後 smoke 只確認鏈路與分數區間，不構成研究有效性證據。

## 現有選型證據（分母與限制不得省略）

| 工作 | 交付來源 | 可支持的結論／限制 |
| --- | --- | --- |
| M4 add 接線 | PR #30；re-enroll PR #32 | 單張真 pipeline 寫入／re-enroll 接線已驗，不是動態掃臉 |
| M6 premise | `d-20260913084623309226-0` | 既有權重與檔名序 B 獲授權；非 EXIF 證實的跨日時間線 |
| M2 Pair 2 | PR #35 | manifest/generation/同協議 bakeoff 已交付；不能由此推論新 30 non-target 的 Pair2 結果已完成 |
| M1 real FA | PR #36 | Pair1、detector 0.8、23 gallery、30 non-target；match 0.45/margin 0.10 時觀察 FA 4/30 |
| M5 二維表 | PR #37 | 7×4 cells；margin 0.15、match 0.30–0.55 時 target 3/13、FA 0/30；match 0.60 時 target 2/13。僅此資料集觀察，不是新產品預設 |
| M3 real replay | PR #38 | 13 個 person-23 探針；baseline matched/review/unknown=2/10/1，adaptive=0/13/0；creation/corroboration/promotion=0，rejections=13 |

M3 的 6 個 correct-supervision 事件最高 score 約 0.6785，均低於 candidate update 0.88；其餘 7 個為 not_me。這證明本串流沒有建立 candidate，**不證明學習增益或長期不污染**。M1 的 30 non-target 未包含在這 13 事件中；不能把 4/30 FA 說成那 7 個 not_me 的子集。Replay 的 retirements/rollbacks 固定 close-out 段須與真實輸入觸發分開，不計為現場生命週期成功證據。

**Contract v1 標註（只標註不重詮釋）：** 上表 M1／M5／M3 皆為 `ALIGN_CONTRACT_VERSION=1` 下的量測。M1/M3/M5 v3 重跑數字已歸檔（另行解讀，不改既有報表）。

**治理機制就位（PR #59）：** `ALIGN_CONTRACT_VERSION` 升版即 `MIGRATION_REQUIRED`；runtime generation `sface-112-rgb+align3`（contract v3）。

Baseline/adaptive 有不同規則；2/13 對 0/13 不是純粹同 operating point 的模型 A/B 勝負。檔名序只為授權的 replay 排序，不代表時間或年齡趨勢。零觀察誤認不代表零真實風險；小 gallery 不外推 500 人。**靜態 M1/M5 成績不推論動態 session 成績，反向亦然。**

原 no-weights [報告骨架](2026-09-12-model-selection-report-skeleton.md) 是歷史證據，不是最新 pending 清單。真實照片、權重、embeddings、DB、可識別逐筆 logs 均留 repo 外；跨 session 應確認外部 artifact 存活與版本，不把 `/tmp` 當永久證據庫。

## 階段與授權，避免重新走錯 gate

1. **1B P0 開工 gate：已完成。** ADR 0006 是 accepted 的分階段架構決策，不是現在等著第一次批准 1B 的待辦。
2. **1B 工程／研究 closeout：** 已交付功能與 M3 真圖負向／閾值 evidence；保留未測的真實 promotion、long-horizon drift 和現場失敗恢復限制。不能僅因 M3 跑完便自動宣布全部 spec acceptance 達成。
3. **選型／operating point：OPEN。** 未決部署準確性不禁止經獨立設計的受控 Mac 研究；研究不降低原有模型 integrity/license/provenance gate。
4. **Phase 2A：已結案（2026-09-16）。** 上述七項交付完成；未驗項 retained（見下），不構成結案阻擋。
5. **Phase 2B：文件（G0）與研究工具工程（G1）已授權；採集（G3）、改善／調參（G4）、holdout 解封（G5）未開。** 範圍、分母契約、分析觸發與完成判定見 [Phase 2B 研究規格](specs/2026-09-16-phase2b-mac-recognition-research.md) 與[執行計畫](plans/2026-09-16-phase2b-mac-recognition-execution-plan.md)；證據隔離取捨見 [ADR 0010](decisions/0010-phase2b-evidence-isolation.md)。工程期間只用 synthetic／test double；任何真實相機操作需 operator 在場。**2B 目前沒有任何辨識有效性證據。**
6. **Android/iOS／認證／產品整合：另行決策。** 代表性 gallery 重校準在部署／目標容量宣稱前完成；跨 runtime 與真人 carrier 在 mobile 評估前完成。相機引導、多幀穩定不等於 liveness。

## 未驗 retained（不隱含派工授權）

- 真機 disconnect 模擬、OS 相機權限層驗證。
- 辨識正確性（屬 2B；工具工程已開工，尚無任何真實 session 證據）。
- 引導 UI pyside6 真窗、§5 對照臂、§8 分母、holdout 切分（屬 2B 工程範圍，未實作）；未註冊參與者 session 另需 G3 採集授權與該參與者個別同意。
- 研究 evidence plumbing 三項已知缺口：attempt 帳本晚於 camera／model setup、`--fixed-seconds` 未實作、replay 以重放時鐘重建 processed（見執行計畫 §10）。修復前，雙臂比較與 session 級分母不得用於任何有效性宣稱。
- R4 缺檔 seq/rank 對位、各臂 refused 計數／雜檔處理、ORT teardown crash、continuity 位移界線初值（0.50 仍為初值）、ORT 端到端 5fps 餘量。

## Next Session

1. 讀本檔、母規格、ADR 0008／0009、Mac 研究設計；查 git／task／inbox 活源。
2. Phase 2A 已結案；Phase 2B 走 G0–G6 關卡，目前 G0／G1 已開，G3 採集、G4 改善、G5 holdout 未開。每個工程包由 lead 建 board task 後才派工，不從文件直接開工。
3. Plan 規範：相機錄製需個別參與者同意；文件批准不構成同意。
4. 不重開已完成的 P0／2A；不把歷史骨架當現況；不重新派已完成任務。
