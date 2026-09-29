# Project State

更新：2026-09-29。**目前主線是「Mac Demo 修復與驗收」（D0–D4），不再是「繼續擴充 G3 研究工具」。**
證據基準：`b92a2762a4f33a6f1fdd209bc70ad9b3e6b96c9e`（PR #115）。D0 已完成並凍結基準，見
[Mac Demo 修復基準 D0](mac-demo-baseline-d0.md)。**D1–D4 尚未開始**（D1 計時修復、D2 持續預覽與
可取消推論、D3 結果呈現、D4 真機端到端驗收）。本檔更早段落仍保留 Phase 2A／2B 的歷史事實與
未驗限制，那部分是背景，不是目前的主線待辦。啟動 session 時仍須查 live main／board；本檔不是
daemon 工作快照。

## 現場紀錄（此前本檔誤述為「尚無任何真實 session」）

**本機已有真機 session 紀錄**（`~/Downloads/face_sample/_facecore/store/results.csv`，
2026-09-24／09-29，共 29 輪；唯讀，未覆寫）。先前「尚無任何真實 session、尚無辨識有效性證據」
的敘述只適用於 Phase 2A 結案當時，**已過時**：

- 29 輪結果：`invalid_input` 23、`timeout` 6、**`matched` 0**；11 輪有 top1，**全部**為 `enroll-23`
  （score 0.6338–0.6778）。**`invalid_input` 的 23 輪全部只採到 1 張影格**（分佈 `{1: 23}`，
  耗時 5264.2–34558.8 ms）—— 全數如此是首幀延遲吃掉整個 5 秒窗口的強訊號。
- 失敗模式集中於**計時**：`qt_window.py:549` 在 `on_start` 前讀 clock，開相機耗時被算進 5 秒辨識窗口
  （open 在 `controller.py:158` 的 `on_start` 內）；終局多為
  `zero_usable_frames_collected` + `deadline_exceeded`（`session.py:504`）。
  **D0 只定位引用，未修改任何程式碼。**
- 已由合成重現確認：相同三幀在 `start=0` 得 `invalid_input`、在 `start=8304` 得 `matched`
  ——計時設計足以產生現場失敗模式，但**不代表真機已修復**。
- Gallery（註冊組 23 張，身分＝檔名 stem）在 `b92a276` 以真模型重算的 digest
  `e3d77c4c…` 與 29 輪 CSV 每輪的 `gallery_digest` **29/29 一致**，模型與註冊資料未漂移。

**這 29 輪的解讀界線（不可省略）：** `label_kind=uncertain` 只代表 operator 當時按了「錯誤」，
**不是系統已知的正確身分**；`unenrolled` 亦為按鍵標註。因此 0/29 matched **不等於 SFace 準確率 0%**，
11 輪 top1 全為 `enroll-23` 也無法區分正確辨識與誤認。**歷史 G3 採集資料可用於工程故障診斷，
不自動視為有效準確率樣本**；代表性人群準確率仍屬未驗（見下方未驗 retained）。

## 仍未確認（需真機，D0 只列不解）

首幀延遲中 open／read 各占多少（無分段時間量測）、真機 UI 卡頓程度、曝光拒絕是場景還是量測／門檻
問題、現有模型在代表性人群的準確率。完整清單與 D1–D4 對應見
[D0 §11](mac-demo-baseline-d0.md)。真機測試需 operator 在場並另行授權（`--device local` 未授權）。

## 固定基準（D0 凍結，D1–D4 不得偏離）

| 項目 | 值 |
| --- | --- |
| 凍結 HEAD | `b92a276`（PR #115） |
| 環境 | uv 0.10.7／CPython 3.14.7／onnxruntime 1.30.0（**必須經 `uv run`**，系統 `python3` 無本專案） |
| 模型 | YuNet 2023mar `8f2383e4…`／SFace 2021dec FP32 `0ba9fbfa…`（`~/facecore-models`） |
| Profile | `g3-v1`：match 0.363／margin 0.10／required_support 3／timeout 5000ms |
| Gallery | 註冊組 23 張，digest `e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f` |
| 設定 | `~/Downloads/face_sample/_facecore/g3-local.json`（`_facecore` 在此，**不在 repo root**） |
| 啟動 | `scripts/g3-local-test-app.command`（注意：launcher 啟動會 `git pull --ff-only`，影響 D4 可重現性） |
| 不開相機重現 | `python -m facecore.research.cli live --device fake --ui fake --store <tmp>/store --key-dir <tmp>/keys --record-consent --image-consent`（**四個旗標皆必填**，已實測 `matched`；完整可執行命令見 D0 §9） |

Baseline 驗證：`pytest tests/ -q` → **867 passed, 8 skipped**（`b92a276`，offscreen）。

## 歷史：Phase 2A／2B（背景，非現行主線）

證據基準：main `275b6de3a05d8b5650f625cf9c7bd06b14e1bd05`（PR #77）。Phase 2A 結案（decision
`d-20260916002300248295-1`）；Phase 2B 文件與研究工具工程已授權開工（decision
`d-20260916012440460338-3`），**工程鏈 E1–E7 已 merge**（見下「Phase 2B 工程進度」），**但當時尚無任何真實 session、尚無辨識有效性證據**——已實作的是研究工具能力，不是研究結論。（2026-09-29 更正：本機已有 29 輪真機紀錄，見上方「現場紀錄」；該 29 輪仍不構成辨識有效性證據。）啟動 session 時仍須查 live main／board；本檔不是 daemon 工作快照。

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
- **G3 R1（2026-09-24 spec／計畫；實作 2026-09-29 已在 main 落地）：** 修訂 [spec](specs/2026-09-24-g3-local-test-app.md)（Start-gated：選相機不開鏡頭、按 Start 才辨識、終局先關鏡頭清畫面）與[修復計畫](plans/2026-09-24-g3-start-gated-repair-plan.md)已落地；實作隨 PR #114（PR-A）與 #115（PR-B）merge，**已由合成／offscreen 測試驗證，仍未通過真機驗收**——真機端到端屬 D4。

## 功能與 evidence 的界線

已有：decode/detect/align/embed/compare、單幀 policy、加密 SQLite／KeyProvider、identity lifecycle、確認候選與 corroboration/promotion/utility/eviction、rollback/delete/export/import、generation migration、drift、replay、capacity 工具。**Phase 2A 新增**：相機 adapter（真機 device 可用＋`local` 列舉解析）、即時 session 聚合器、研究 session recorder（同意／加密／TTL／可恢復刪除）、加密存幀與逐幀 ID ledger、研究 CLI（live/replay/delete）。可用 CLI 語法以 `facecore.sh --help` 與目前 parser 為準；母規格的示意 `identify --confirm-learning` 不等於所有使用情境已有端到端產品 UI。

尚未有：既有人工照片驗收及 synthetic 功能測試，不保證換造型、跨日學習、500 人辨識或相機防翻拍效果。**真機辨識正確性未驗證**——修復後 smoke 只確認鏈路與分數區間，不構成研究有效性證據；本機 29 輪真機紀錄同樣只證明執行過、**未證明辨對**（見上方「現場紀錄」解讀界線）。真機端到端驗收屬 D4，尚未執行。

## Phase 2B 工程進度（G1 範圍；能力已落地，證據仍為零）

下列為 `275b6de3` 已 merge 的研究工具能力。**全部只用 synthetic／test double 驗證**：能力存在不等於已用真人 session 驗證，更不等於辨識有效。

| 包 | PR | 落地能力 |
| --- | --- | --- |
| E1 | #71 | 研究嘗試帳本前置於 camera／model setup（`research/cli.py:353 recorder.begin_attempt`） |
| E2 | #72 | 既有量測點補診斷事件原因，未另立第二條 pipeline |
| E3 | #73 | 固定窗口採集 `--fixed-seconds`（`research/cli.py:1162`、`:698`／`:711`）＋兩個真入口接線 |
| E4 | #74 | replay 以**原時間**重演＋雙臂評估（`research/replay.py` `ArmOutcome:78`／`evaluate_arms`） |
| E5 | #75 | attempt-aware 分母、診斷分類與每批分析入口（`research/analysis.py:391 analyze_batch`） |
| E6 | #76 | prospective holdout candidate freeze（`research/split.py` `CandidateFreeze`／`HoldoutRelease`） |
| E7-B | #77 | 最小 pyside6 引導真窗＋方形採集幾何契約（`live/qt_window.py`，spec 附錄 A） |

`report.py:summarize()` 的舊分母與 `analyze_batch` 的新 attempt-aware 分母**刻意並存**（執行計畫 `:301`／`:461` 明文要求保留舊 caller 相容性，不以新分母重新詮釋舊 M3 報表）；這不是待統一的技術債。

**仍不可宣稱**：`TOOLING_READY`（屬 G2，且真機 smoke 未做）、採集就緒、任何準確率。Qt 真窗僅以 offscreen 驗證，**offscreen 不構成 macOS 相機權限層或真實裝置的證據**。

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

- 真機 disconnect 模擬、OS 相機權限層驗證（2A 移交）。E8-A 的全鏈 e2e 為 synthetic／offscreen，**不涵蓋也不替代**這兩項；E8-B 真機 smoke 需 operator 在場並另行授權。
- 辨識正確性（屬 2B；工具工程已開工）。2026-09-29 更正：本機已有 29 輪真機紀錄，但標註為按鍵推記
  （`uncertain`／`unenrolled`），**不構成辨識有效性證據**；代表性人群準確率仍未驗（見上方「現場紀錄」解讀界線）。
- 引導 UI pyside6 真窗、§5 對照臂、§8 分母、holdout 切分之**工程能力已於 E3–E7 落地**（見「Phase 2B 工程進度」），但**全部僅 synthetic 驗證**；未註冊參與者 session 另需 G3 採集授權與該參與者個別同意。
- 研究 evidence plumbing 三項缺口**已修復**（E1 attempt 帳本前置、E3 `--fixed-seconds`、E4 replay 原時間重演）。修復本身只解除工具面阻擋；本機已有 29 輪真機紀錄，但因標註為按鍵推記，雙臂比較與 session 級分母**仍不得用於任何有效性宣稱**。
- R4 缺檔 seq/rank 對位、各臂 refused 計數／雜檔處理、ORT teardown crash、continuity 位移界線初值（0.50 仍為初值）、ORT 端到端 5fps 餘量。
- **E8-B fixed-window 契約（decision `d-20260920132145277296-1`，Issue #84 open）：** 正式 profile 改為 5 秒／相鄰樣本至少 200ms／至多 26 張（含 t=0 與 t=5000 兩端點），跨欄位必要條件 `max_frames >= ceil(timeout_ms / sample_interval_ms) + 1`；保留 t=0 立即取樣、5000ms deadline、deadline_reached 與 paired full 語意。#84 為 E8-B 重跑前 blocker。**此為 spec landing，不等於 production implementation go**——production follow-up 須另提 implementation plan 並再次取得 operator go。

## Next Session

1. 讀本檔、[D0 基準](mac-demo-baseline-d0.md)、母規格、ADR 0008／0009；查 git／task／inbox 活源。
2. **現行主線是 Mac Demo 修復與驗收 D0–D4。** D0 已完成（凍結基準見 D0 檔）；D1（相機啟動與辨識窗口
   計時修復）是下一個工作包，須由 lead 建 board task 後才派工，**不從文件直接開工**。D2–D4 依序為
   持續預覽與可取消推論、結果呈現、真機端到端驗收。
3. D1–D4 不得偏離 D0 凍結基準（版本、profile g3-v1 門檻 0.363／0.10、gallery 23 張），不得為得到
   matched 而直接降門檻或移除三幀確認。
4. Plan 規範：相機錄製需個別參與者同意；文件批准不構成同意。真機測試需 operator 在場並另行授權。
5. Phase 2A 已結案、G3 暫停擴充採集；不重開已完成的 P0／2A，不把歷史骨架當現況，不重新派已完成任務。
