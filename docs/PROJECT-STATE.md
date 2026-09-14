# Project State

更新：2026-09-14。證據基準：PR #38 merge `ff923a3a1bc9fdc329a306e87146279cc0290f69`。啟動 session 時仍須查 live main／board；本檔不是 daemon 工作快照。

## 現在在哪裡

- Phase-1A 靜態 pipeline／CLI／Layer-A conformance 已交付。
- Phase-1B plan 的 PR-C–G（Task 1–11）及後續接線／治理修正已 merge；不是「尚未取得開工 go」。P0 已於 decision `d-20260912041106780391-34` 放行，實作全波段授權為 `d-20260912090115587658-42`。
- 真圖 replay 已在 M3／PR #38 接入報表並重跑；不是「只有 synthetic／等待權重」。但真實 candidate creation/promotion 成效及長期 drift 尚未證明，不能宣稱治理已獲全面真實情境驗證。
- SFace Pair 1 維持 provisional，**selection gate OPEN**。M1–M6 工作交付不等於選型條件全部滿足，更不等於安全認證或正式部署核准。
- Operator 已同意先整理 **Phase 2A Mac 動態辨識研究原型** 設計與更新文件；[正式研究設計](specs/2026-09-14-mac-live-identification-research-design.md)／[ADR 0008](decisions/0008-mac-live-identification-research.md) 為新研究邊界；[實作計畫](plans/2026-09-14-mac-live-identification-implementation-plan.md) 已於 PR #40 merge。S1/S2 spike 與 D1 選型凍結已完成，T1 契約與 T2 單幀管線已完成交付，T3 有界 session 引擎已完成實作並提出 continuity 初值 0.50。相機錄製與 mobile 尚未授權。

## 功能與 evidence 的界線

已有：decode/detect/align/embed/compare、單幀 policy、加密 SQLite／KeyProvider、identity lifecycle、確認候選與 corroboration/promotion/utility/eviction、rollback/delete/export/import、generation migration、drift、replay、capacity 工具。可用 CLI 語法以 `facecore.sh --help` 與目前 parser 為準；母規格的示意 `identify --confirm-learning` 不等於所有使用情境已有端到端產品 UI。

尚未有：相機 adapter、即時 session 聚合器、引導 UI、研究 session recorder；既有 replay harness 不替代它們。既有人工照片驗收及 synthetic 功能測試，不保證換造型、跨日學習、500 人辨識或相機防翻拍效果。

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

Baseline/adaptive 有不同規則；2/13 對 0/13 不是純粹同 operating point 的模型 A/B 勝負。檔名序只為授權的 replay 排序，不代表時間或年齡趨勢。零觀察誤認不代表零真實風險；小 gallery 不外推 500 人。

原 no-weights [報告骨架](2026-09-12-model-selection-report-skeleton.md) 是歷史證據，不是最新 pending 清單。真實照片、權重、embeddings、DB、可識別逐筆 logs 均留 repo 外；跨 session 應確認外部 artifact 存活與版本，不把 `/tmp` 當永久證據庫。

## 階段與授權，避免重新走錯 gate

1. **1B P0 開工 gate：已完成。** ADR 0006 是 accepted 的分階段架構決策，不是現在等著第一次批准 1B 的待辦。
2. **1B 工程／研究 closeout：** 已交付功能與 M3 真圖負向／閾值 evidence；保留未測的真實 promotion、long-horizon drift 和現場失敗恢復限制。不能僅因 M3 跑完便自動宣布全部 spec acceptance 達成。
3. **選型／operating point：OPEN。** 未決部署準確性不禁止經獨立設計的受控 Mac 研究；研究不降低原有模型 integrity/license/provenance gate。
4. **Phase 2A 研究：設計文件階段。** 下一步是書面設計 review/merge、implementation plan、operator 實作 go。錄製需個別參與者同意；文件批准不構成同意。
5. **Android/iOS／認證／產品整合：另行決策。** 代表性 gallery 重校準在部署／目標容量宣稱前完成；跨 runtime 與真人 carrier 在 mobile 評估前完成。相機引導、多幀穩定不等於 liveness。

## 後續範圍

研究第一版固定 one-shot gallery，以 session 為評估單位；保存採明確同意、加密、TTL／刪除，詳細契約只在研究設計中維護。未註冊者、本人認錯身份、timeout 和品質失敗都要計數；先蒐集小批 session，再封存 holdout 比較，無須先手工整理大圖庫。

後續獨立研究：受控離線學習 replay（包含 unknown 與錯認事件）、更代表性圖庫、模型對比、現場效能與處理失敗。待驗項：R4 缺檔 seq/rank 對位、各臂 refused 計數／雜檔處理、ORT teardown crash（不可類比 timing flake）、continuity 位移界線初值、ORT 端到端 5fps 餘量、pyside6／Cocoa 最終抉擇。這些是已知限制／後續範圍，不隱含派工授權。

## Next Session

1. 讀本檔、母規格、ADR 0008、Mac 研究設計；查 git／task／inbox 活源。
2. T2 單幀管線（PR #45 `c7ee145`）已 merge。T3 有界 session 與兩策略已完成交付，並完成 D1 §11.3 continuity 位移界線初值（0.50）與命名測試。T5 同意加密 TTL 可恢復刪除已完成交付（ResearchRecorder＋ResearchKeyProvider，僅 synthetic 驗證）。T4 capture pump／controller 已完成交付（CaptureSource＋LiveController，latest-slot1，drop 端到端非零可達，S1 camera-free 重跑全 PASS）。T6 回放 ground truth 隔離與報告已完成交付（replay_session＋summarize，含 T5 N1 dims cap，僅 synthetic 驗證）。下一步依 Lead 派工（T7 本機視窗與研究 CLI）。
3. Plan 規範：T1–T8 依序推進，未完成 T7/T8 前不寫正式 GUI 入口。相機錄製需個別參與者同意；文件批准不構成同意。
4. 不重開已完成的 P0；不把歷史骨架的缺權重、未跑真圖或本機舊 main 當現況；不重新派 M1/M2/M3/M5 已完成任務。
