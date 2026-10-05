# Project State

更新：2026-10-03（同步至 `main` `2059380`）。**目前主線是「Mac Demo 修復與驗收」（D0–D4）＋「D5／D6 靜態與模型比較」，並正在推進「D7-A 診斷 log 基礎設施」；不再是「繼續擴充 G3 研究工具」。**
證據基準：`cf1d6dd0b3f4238ce333f961b74efa24c9e5b820`（PR-A，D5 harness）。D0 已完成並凍結基準，見
[Mac Demo 修復基準 D0](mac-demo-baseline-d0.md)。

**D0–D4 已全部完成並合併**（PR #119／#120／#121／#122／#124／#125）。**D5 的 Task 1（harness ＋ 計數語意，
PR-A）與 Task 3（baseline 報告，PR-B）亦已交付**，報告見
[D5 現版本靜態 baseline](research/2026-09-30-d5-static-baseline.md)。

**其後續合併（2026-10-01 起）** —— 讀本檔時請一併知道，本檔其後段落仍保留較早的狀態敘述：

| 合併 | PR | 帶來什麼 |
| --- | --- | --- |
| D6 spec §7 三層定案 | #137 | 刪除「衝突時以 plan 為準」，改為 spec／plan／decision 三層各司其職 |
| D6 R 報告 | #136 | Facenet512 vs SFace control，兩臂各 43 列 |
| **D7-A W3** | #138 | demo log 由 23 欄擴為 **35 欄**（第 24–35 欄為診斷欄位） |
| **D7-A W0-a** | #139 | 診斷 run 操作手冊（operator 可不透過 agent 獨立執行），見 [runbook](w0a-diagnostic-run-runbook.md) |
| **D7-A W1** | #142 | demo log 再加 **3 欄 → 共 38 欄**（第 36–38 欄 `expected_count`／`loaded_count`／`gallery_rejected`） |

⚠️ **demo log 現為 38 欄。** 欄位常數在 `research/cli.py` 的 `G3_DEMO_RESULTS_CSV_COLUMNS`；
既有 23 欄位置**不得**變動（operator 試算表公式依賴 `margin` 第 18 欄與 `label_kind` 第 22 欄）。

D4 真機第一輪的觀察與 D5 的靜態結果分屬不同協議（見 D0 §11 第 17、18 項與 D5 報告第 8 節），
**不得互相推論**。本檔更早段落仍保留 Phase 2A／2B 的歷史事實與未驗限制，那部分是背景，不是
目前的主線待辦。啟動 session 時仍須查 live main／board；本檔不是 daemon 工作快照。

## D5 現況（2026-09-30）

- **H1 成立**：以 `_build_true_context` 建出的 gallery digest
  `e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f` 與 D0 凍結值逐字相符，
  **報告與 App 現場評分同一個註冊組**。
- **13 張辨識組**：L 分支 12/13 正確接受、**0 認錯人**；R 分支 13/13。唯一落差是 1 張品質拒絕。
- **30 張 non-target**：L 分支 **FA 2/30**（現行門檻 0.363／0.10）、review 6/30、
  **品質拒絕 22/30**、正確拒絕 0/30。R 分支 review 28/30、FA 2/30、**品質拒絕 0/30**
  （R 跳過品質篩選，那個 0 是量測結果不是缺資料）。
- **主導結果是品質篩選，不是模型**：22/30 的 non-target 在品質關卡就被擋掉，剩下 8 張裡 6 張落在
  review。**L 分支下沒有任何一張 non-target 是被模型判為 unknown 的。**
- **品質拒絕 23/23 全部是 `quality_exposure`**（該 code 同時涵蓋過暗與過曝）。
  **D5 讓 D0 §11 第 3 項更清楚，但沒有回答它** —— 單一批次、單一場景、無光照對照，
  照片偏暗與量測誤判無法從這批資料分辨。
- **門檻未被選定**：sweep grid 以 live 語意呈現 28 格，`selected` 全部為 False。
  **這批照片沒有 holdout，事後挑門檻等於用測試集調參。**
- **D6（換模型後端）已完成**（R 報告 #136、spec 定案 #137；A3 本機 run 2026-10-01）—— 見下方「D6 現況」。

## 現場紀錄（此前本檔誤述為「尚無任何真實 session」）

⚠️ **2026-10-03 operator 已刪除 `~/Downloads/face_sample/_facecore/store/demo-results.csv`**。該檔是**一份 32 列、23 欄、因 header drift 而錯位的舊檔**（見 issue **#140**；當時程式只寫得出 23 欄，W3／W1 之後才擴到 35／38 欄）—— ⚠️ **那組列數／欄數只描述「該檔在被刪除之前是什麼形狀」，檔案本身已不存在，不得當成現存的檔案特徵引用。** ⚠️ **因此本檔下方所有指向「那批 32 列 log」的數字，描述的是「當時發生了什麼」，不是「檔案現在還在」** —— 事實本身不因砍檔而失效，引用時須註明資料檔已刪。

⚠️ **同一日 operator 又重新跑出新的 `demo-results.csv`（38 欄，內容持續增加中）**，隨後 PR #145（#141 丙-新）合併後 **App 改為每次啟動開一個新檔 `demo-results-<啟動時間>.csv`**，所以那個固定檔名**不再會被產生**。⚠️ **`tests/cli/test_d7_w3_diagnostic_log_fields.py` 的 `TestDocumentedScopeMatchesTheLog` 已不再讀 operator 的 CSV** —— 舊的「讀固定檔名、斷言實測值」寫法已被守護者取代：`RETIRED_FIGURES = (16, 11, 122, 32)`，`assert not (restated and not cited)`（註解重述已刪批次數字且無 commit 引用才紅）。**operator 重跑 App 不會使該測試轉紅**，因其輸入是 repo 內的 `cli.py` 註解，不是 repo 外的可變檔案。

**demo log 的存放位置是 `store/` 目錄**（在 repo 外，唯讀，**不得把逐列原始資料抄進 repo**）。⚠️ **不要依賴某一份檔案是「唯一」的那份** —— 實測該目錄下同時存在固定檔名與 `demo-results-<啟動時間>.csv` 兩種產物，operator 重跑會持續新增；**要問「當時有哪一份」就去列目錄，不要引用本檔的列舉**。

⚠️ **例外：`~/Downloads/face_sample/_facecore/enroll-24-cross-identity-baseline.csv` 不在 `store/` 裡**（它在 `_facecore/` 根目錄），是 operator 另跑出、另存的一份跨身分基準檔，本節以下逐列記錄的是它：

該檔（1341 bytes）2 列：

- r1 `invalid_input`／`all_frames_rejected_no_face|deadline_exceeded`，sampled 25／usable 0／rejected 24，證據段 5175ms
- r3 `matched`，top1 `enroll-24` @ 0.5226、top2 `enroll-17` @ 0.3206、margin 0.2021，sampled 23／**usable 3**／rejected 20，證據段 4895ms

⚠️ **該檔是 35 欄格式，不是最新的 38 欄** —— 它產生於 W1（#142）合併之前，**沒有** `expected_count`／`loaded_count`／`gallery_rejected` 三欄。operator 下次跑出來的檔會是 38 欄。

⚠️ **這是 D7-A 以來第一筆跨身分 ground truth**：operator 新增 `enroll-24` 並成功辨識、按「正確」標註（`label_kind=enrolled`），**但樣本只有 2 列、單一身分、不含 non-target**。**它不足以回答「門檻該不該調」或「會不會認錯人」** —— 那需要 W0-b 的跨身分＋非目標輪替 run，而 W0-b 是 operator 的 key issue、**尚未排程**。

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
| 啟動 | `scripts/g3-local-test-app.command`（**D4 前置 PR 後：launcher 只顯示 commit 與落後狀態，不再 `git pull`** —— 版本由 operator 決定，固定版本驗收因此可重現） |
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
| D6 A3 候選比較 | R 報告 | 兩臂各 43 列、gallery 23/23、同 SFace bytes bridge 成立；**probe 身分為單一 `enroll-23`：M3 的 13 個探針（`d5-static-baseline-plan.md:39`：`person-23` ＝ live gallery 的 `enroll-23`）與本輪 13 張皆同；歷史 29 輪現場 session（**同一 stream，非獨立 probe 量測**）中 11 輪有 top1 亦全為 `enroll-23`。不支撐跨身分判別力**；Facenet512 分數上偏致 SFace 門檻不可沿用。**未選門檻。** |

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

## D6 現況（2026-09-30）

D6 是「用相同 23／13／30 照片集合、固定 detector 與 alignment，只換辨識 backend」的模型比較，
**不改現有 App，不是產品模型替換、不是門檻選定、不是部署授權。**

- 協議與計畫：[D6 功能協議](specs/2026-09-30-d6-deepface-comparison.md)、
  [D6 計畫](plans/2026-09-30-d6-deepface-comparison-plan.md)。
- 候選資格與未解事項：[D6 候選 gate](research/2026-09-30-deepface-candidate-gate.md)。
- **可用候選四個**：`ort-sface-control`（既有凍結 artifact）、`deepface-sface-bridge`（同一份已核 SFace
  weight bytes，價值在抓色彩／尺度／前處理接錯），以及 **`deepface-facenet512`／`deepface-arcface`** ——
  operator 於 `d-20260930132705762549-32`（裁決 B）**核准限本機研究比較**下載與使用。
- **核准的邊界**：限本機研究比較 —— 不進 Git、不散布、不商用、不進產品 runtime；repo 為 public，只
  commit 授權審查文件，**不 commit 權重檔本身**。**核准使用不等於 license 已證實**：兩個 H5 的 exact
  license／provenance **仍未證實**，operator 接受此風險，D6 報告必須明確記錄。
- **下載前審查**：downloaded-to-cache-then-hash、no-network-at-inference；actual SHA 不符 registry 立即
  fail，不得改 expected 值湊數。
- 兩份候選 gate（2026-09-10 的產品 bake-off 與本輪的 D6 候選）**並列不合併**；D6 的新 gate 不回頭
  改寫舊檔，理由記在該檔開頭。
- **A3 本機 run 已完成（2026-10-01）**，結果與限制見
  [D6 比較報告](research/2026-09-30-d6-deepface-comparison.md)。指標摘要：
  同 SFace bytes bridge 成立（cosine median 1.0、maxdiff median 2.4e-07）；
  兩臂各 43 列、unprocessable 0、gallery 23/23；**top1 正確 k／13 在 SFace control
  與 Facenet512 都是 13/13**。
  **這兩列不可當模型等價的證據 —— probe 13 張全部是單一身分 `enroll-23`。**
  Facenet512 的 probe 與 non-target 分數同步偏高（non-target 有 9/30 高於最低
  probe 分數，control 為 2/30），**故不可套用 SFace 門檻**。
  **未選門檻、未做 frontier、未做獨立校準。**
  ArcFace 已核准且 artifact 已核，但架構建構階段與該 Keras 組合不相容，
  **環境本身可行**（Facenet512 在同一環境完成量測），依 plan `:86` 屬
  不預設必跑的第二候選，**交付不受影響**。
- 目前階段是 A 系列已交付、**R 彙總報告已產出**；**這不是 shortage 結案，也不是
  產品門檻選定或部署授權** —— plan §11 的完成定義另需獨立校準與新資料驗證。
- 所有 `selected=False`；D6 不選產品門檻。

## Next Session

1. 讀本檔、[D0 基準](mac-demo-baseline-d0.md)、母規格、ADR 0008／0009；**若要操作本機 App，另讀 [runbook](w0a-diagnostic-run-runbook.md)**；查 git／task／inbox 活源。
2. **現行主線**：
   - **D0–D4 已完成**；**D5 靜態 baseline 已交付**；**D6 已完成**（PR #136 R 報告、#137 spec 定案；A3 本機 run 2026-10-01 完成，見下方「D6 現況」）。
   - **D7-A 診斷 log 基礎設施進行中**：W3（#138）、W0-a runbook（#139）、W1 gallery 可見性（#142）已合併。**demo log 現為 38 欄。**
   - **D4 真機驗收仍未完成**（board task open，**19 項**待驗，須 operator 在場 —— 見 [SOP D0 §11 對照表](g3-local-test-sop.md)）。
3. ⚠️ **兩筆 open issue，裁決權在 operator，不在 agent**：
   - **#140（bug，已關閉）**：demo CSV header 不隨欄位擴充更新，既有 store 的新列以欄名讀取回傳 None。**已由 PR #145 修掉（裁決見 decision `d-20261003071803476368-1`，採丙-新）：寫入前比對完整 header tuple，不符則 refuse 並零寫入；另加 per-App-execution 檔名，既有 CSV 永遠不需升級 header。** issue 狀態 CLOSED（`2026-10-04T02:03:07Z`，operator 關閉）。
   - **#141（decision，已裁決）**：demo 診斷 log 格式重新設計，裁決結果採**丙-新**（decision `d-20261003071803476368-1`）：**維持 CSV 為 operator 主路徑**，加寫入前完整 header tuple 比對與 per-App-execution 檔名，**明確排除 sqlite**。issue 狀態 CLOSED（`2026-10-04T02:03:32Z`，operator 關閉）。**是被裁決了，不是被駁回。**
4. **不得把 D7-B（模型選擇）寫成已授權 —— 實際上 D7-B 已取消**（decision `d-20261004021722433077-0`：operator「SFace 就是最終選擇」，後續準確度工作皆為校正，不做模型比較）。原前提（W0-b 產出跨身分資料）已不存在，**本條禁令因此不適用，但禁令文字保留** —— 它是給未來讀者的保護，「不適用」與「不存在」效果完全相反。⚠️ **不得因 W0-b 未排程而阻擋 W2／W5／W4**；計畫順序為 W3 → W0-a → W1 → W2 → W5 → W4，**W4 必須最後做**（它消費 W3 的欄位形狀）。
5. **授權邊界以 decision board 為準，不以本檔為準** —— 本檔若與較新的 decision 衝突，以 decision 為準並修正本檔。
6. D0–D5 不得偏離 D0 凍結基準（版本、profile g3-v1 門檻 0.363／0.10、gallery 23 張），不得為得到 matched 而直接降門檻或移除三幀確認。**D5／D6 未選定新門檻**——sweep 只作呈現。
7. **不得把 D5 的靜態結果推論到動態**（三幀規則不適用於單張照片），也**不得把 M1／M5／v3 的歷史 FA 數字與 D5／D6 並列比較**（gate、contract、gallery digest 都不同）。
8. Plan 規範：相機錄製需個別參與者同意；文件批准不構成同意。真機測試需 operator 在場並另行授權。
9. Phase 2A 已結案、G3 暫停擴充採集；不重開已完成的 P0／2A，不把歷史骨架當現況，不重新派已完成任務。
