# G3 Start 控制相機與真機接線修復計畫

**Source of truth:** `docs/specs/2026-09-24-g3-local-test-app.md` 的修訂草稿（commander workspace `spec-landing/docs/specs/2026-09-24-g3-local-test-app-r1.proposed.md`，目標是取代原檔，非第二份功能權威）；方向裁決 `d-20260924080117997753-5`；RCA board `t-20260924074541870720-80684-50`。
**Base:** `cheerc/portable-face-core` main `6b058fd8335581bba644850f91b93a381482390c`；lead 開工時核新 HEAD，不能把此 SHA 當永遠最新。
**Goal:** operator 選相機不開鏡頭；按 Start 才開、持續預覽與辨識；終局先關鏡頭並清除照片，只留文字結果；標註成功後保留選相機、畫面清空，下一輪需再按 Start。修復真機預覽停格、倒數不動、提早 `zero_usable_frames_collected`。
**Architecture:** 復用現有 `cmd_live` → `DesktopSession`／`SessionEngine` → Qt／`ResearchRecorder`。將各輪的相機、裁切、預覽、trace、影像與時計責任綁回各輪 session，不建立第二條 scorer/policy pipeline；Qt 僅管理 Ready／Running／Result 與資源釋放。
**Tech stack:** Python 3.14、PySide6 6.11.2、OpenCV（既有 capture）、pytest、ruff、mypy；所有 agent 測試 camera-free，operator 最後自行真機測。

## Global Constraints

- 不改 SFace/YuNet 權重、profile `g3-v1`、門檻、gallery、fixed-window 研究路徑、headless/checkpoint 語意、learning；不引入多臉方案或新同意帳本。
- 不讀真人 store、未授權不開相機、不把真人截圖或可識別逐輪資料放 Git／issue／PR；issue 若需證據僅列去識別 UI 文字與 synthetic 重現。
- 已完成的加密輪次／CSV 不刪改；新輪次資料與 operator 標註必須即時、逐輪一致。發生資料寫入失敗就停在結果、不得顯示「已保存」或默默開始下一輪。
- commander 只交 spec／plan／RCA，不 mutate repo、不 commit、不開 PR、不派 reviewer、不 merge；lead 負責實作收斂與最後 SOP 落地。

## Evidence and Open Hypotheses

**Established（來源與行為分開）：**
- operator 截圖有一幀預覽、`invalid_input：zero_usable_frames_collected`、倒數 `5000 ms`；只代表當時沒有合格觀察，不代表相機從未讀到影格。圖片不傳給其他 agent。
- 初始 desktop 帶 `frame_sink`／`frame_transform`（`src/facecore/research/cli.py:988-1000`），連續 round desktop 漏帶（`:1122-1134`）；running 時 standby timer 停止（`src/facecore/live/qt_window.py:509-513`），因此後續影格缺預覽路徑，且未套 square crop／未按輪存幀。
- `_square_capture_transform` 綁初始 `resolved_attempt_id`（`cli.py:961-970`）；`ResearchRecorder.append_frame` 只接受**恰一個** active session（`research/recorder.py:269-277`），初始與 round 的 begin 生命週期須在接線前釐清。
- Qt 倒數 closure 讀初始 desktop（`cli.py:1011-1028`），round 卻是新 desktop；`qt_window.py:535` 每 tick 呼叫 `run_until_terminal(max_steps=50)`，非 fixed 模式若未 terminal，`desktop.py:187-200` 立刻 `finish`。`session.py:237-264, 498-505` 將全拒絕影格歸為 `zero_usable_frames_collected`。合成輸入重現此結果（sampled=1／usable=0），**真機的具體拒絕原因仍未知**。

**Unproven（不可冒稱已驗）：** 真機是否還有 detector/quality/取樣瓶頸。owner＝lead；僅用 synthetic trace 驗接線與分類，最後由 operator 真人測後回報欄位值；不得為解猜測而讀真人影像或調門檻。已知真機上的所有 failure 是否只由上述三處修復也未知；第一輪修正後若仍失敗，以新的非生物診斷再分流，不事先擴大 scope。

## Documentation impact check — `arch=N adr=N area=Y`

`docs/specs/2026-09-24-g3-local-test-app.md:10,23-30,69` 仍明訂選相機後自動待機／人臉觸發／相機不重開，與 operator 新裁決相反；`docs/g3-local-test-sop.md:19-25` 也指示選相機即預覽、無需按 Start。這是 G3 **area** 文件變更，不改 Face Core 架構或新 ADR。**blocking dependency D0：lead 先執行 `project-docs-maintain` check、將修訂 spec（以及 `docs/PROJECT-STATE.md` 必要的授權／pointer 更正）依 docs 流程落地，明列新行為尚未實作；實作 PR 隨行更新 SOP，避免文件提前教 operator 使用未落地的流程。** `docs-check` 結論與受影響文件記在 D0 task；若 check 發現額外 arch／adr 影響，先停並回 operator，不按此計畫盲做。

## File／Interface／Dependency Map

| 位置 | 既有責任／本輪範圍 |
|---|---|
| `src/facecore/research/cli.py:582-711, 948-1159, 1298-1390` | `cmd_live` 啟動、每輪 factory、preview／crop closure、recorder 與 Qt 時間 callback、close-out；G3 真入口要帶 per-round context，但保留其他 caller。 |
| `src/facecore/live/qt_window.py:392-407, 475-555, 571-690, 710-768, 835-913` | 相機選擇、Start／timer／Result／標註、預覽 pixmap 與 close。 |
| `src/facecore/live/desktop.py:150-213, 338-355`、`src/facecore/live/controller.py:145-178, 203-314, 372-425, 644-...` | `start_session`、`run_until_terminal`／步數與 deadline、release source／worker；必要時只改 G3 caller 或加入不破壞 headless 的明確步進 API。 |
| `src/facecore/research/recorder.py:237-309` | begin／append_frame 的單一 active 約束；僅在 D1 spike 證明既有 API 無法正確串接時才修改，否則保留。 |
| `tests/live/test_g3_start_gated_flow.py`（新增）、`tests/live/test_g3_qt_continuous.py`、`tests/live/test_g3_commit_on_label.py`、`tests/live/test_qt_window.py`、`tests/research/test_cli_lifecycle.py` | 以真正 G3 CLI 建立 round factory，僅在最底層替換相機與合成 scorer；測狀態、clock、影像及各輪關聯，回歸舊研究模式。 |
| `.github/workflows/ci.yml` | 現有 qt-smoke 只跑 `test_qt_window.py`＋一條 lifecycle 測試（原檔 qt-smoke job）；把新整合測試加入 macOS／Ubuntu offscreen job，不能只靠本機 GREEN。 |
| `docs/specs/2026-09-24-g3-local-test-app.md`、`docs/PROJECT-STATE.md`、`docs/g3-local-test-sop.md` | spec D0 落地；SOP 與行為 PR 同步改為 Start 才開，PROJECT-STATE 不把未做的實作寫成已驗。 |

**Existing interfaces:** `cmd_live(..., continuous, ui, capture_factory, config, ...)`、`QtResearchWindow(..., next_session, results_csv, ...)`、`DesktopSession(..., frame_sink, frame_transform, trace_recorder, trace_attempt_id, release_source_on_terminal)`、`ResearchRecorder.begin(session_id, consent)／append_frame(frame)／commit(result)`。每輪 id、mapping、frame／trace 與 labels 如何共享 context 是 D1 要驗的**實際接口問題**，不能先寫成已存在的新 signature。

**資料／權限／migration：** 無新身份資料 schema 或授權；舊資料不重算、不改 metadata；Start 前不能建立影像 bundle／讀取鏡頭。operator 使用已同意的本機 store；fail-closed 錯誤後可重新 Start。Rollback 僅 revert 新 PR，不動已留存輪次。真機權限由 operator 在 Terminal 系統彈窗掌控。

## D0 — 文件影響與 spec 落地（blocking）

**Owner:** lead。
**Files:** replace `docs/specs/2026-09-24-g3-local-test-app.md` using commander staged `spec-landing/docs/specs/2026-09-24-g3-local-test-app-r1.proposed.md`（來源位於 commander workspace，目標檔名不含 proposed）；`docs/PROJECT-STATE.md` 依 docs check 加最短當前狀態 pointer。
**Acceptance:** `project-docs-maintain` 產出 docs-check `arch=N adr=N area=Y`；spec 的自動待機規則確實被 Start-gated 規則取代；文件標「實作待交付」，不宣稱已真機驗證；lead 走正常 review／CI／PR。SOP 先保留舊操作直至程式 PR merge，屆時同 PR 改掉。
**Verification:** `git diff --check`、`rg -n '待機.*預覽|偵測到人臉即開始|選定後應顯示.*預覽' docs/specs/2026-09-24-g3-local-test-app.md` → 不命中舊行為；核對最新 spec 正文與 staged source identity。D0 merged SHA 才進 D1／實作。

## D1 — 分輪接線 spike（analysis-only，time-box 45 分鐘）

**Owner:** lead 指定 implementer；獨立 board task，依賴 D0。只讀 source；可在 daemon-bound worktree 以 synthetic 假相機／tmp store 做可丟棄的最小重現，**不改產品碼**。
**Question:** `cmd_live` 真正 picker 分支如何以最低葉節點換 FakeCapture（直接傳 `capture_factory` 會走不同分支）；初始 session 何時清理才能維持 `append_frame` 單 active；哪個 per-round context 同時擁有 `attempt_id`、`session_id`、frame_sink、square transform、診斷與 label；clock／source-open 從哪一輪取值；Qt 非 fixed 的 50-step 如何只步進不提前 finish。
**Evidence:** 列出各分支對照 `source.read → scorer → frame_sink → recorder → preview` 與 `Start → open → terminal → release → label` 的 producer／consumer；跑 synthetic 最小重現並保存 command／輸出（不能把真人畫面或路徑寫入 board）。對每項前提標 confirmed/refuted、指出最小 interface 變更與回歸面。
**Stop:** 45 分鐘內若 single-active 或資料所有權無法在現有介面內解釋，報 `BLOCKED`＋一個需要 lead 裁定的最小方案，不先派 implementation；不把未知 API 名稱填作本 plan 的事實。D1 報告供後續 PR brief 凍結 exact signatures。

## PR-A — 每輪資料與計時正確（依賴 D1）

**Goal:** 保持目前 G3 入口可用時，先補 per-round preview/crop/staging/trace/clock，與「尚未到期限不得由 50 steps 提早回 invalid_input」；不先改相機選擇與 Start 操作方式。若 D1 證明無法在舊 UX 下獨立交付，lead 將此 PR 與 PR-B 合併，並在 board 記錄為何單獨 merge 會留下 broken state。
**Files:** `src/facecore/research/cli.py`、必要時 `src/facecore/live/desktop.py`／`controller.py`／`research/recorder.py`；新增 `tests/live/test_g3_start_gated_flow.py` 中的 factory／frame／clock 子測試，保留既有 G3 測試。
**Consumes:** D1 的每輪所有權與入口圖；現存 `cmd_live`、`DesktopSession`、`ResearchRecorder` 接口。
**Produces:** 每輪具可證的 preview／square crop／frame staging／trace ownership；本輪時鐘；多次 UI tick 可持續至 matched 或真的 deadline。若需新增接口由 D1 manifest 提出後寫進 PR brief，不在此先造 symbol。
**Test-first evidence:** 新 synthetic regression 經真 `cmd_live` Qt picker 分支（只替換 OpenCVCapture／模型葉節點）驗一輪不同影格預覽、crop mapping、加密 frame 與 attempt/session 關聯。修前應可觀察「round preview 不變／no staged frame／錯誤 id」；另一 case 50 步只有低品質幀、時間未滿 5 秒，修前提早 `invalid_input`，修後繼續到實際 deadline。
**RED/GREEN:** `QT_QPA_PLATFORM=offscreen uv run --extra dev --extra research-ui python -m pytest tests/live/test_g3_start_gated_flow.py -q` → RED 命中新測試斷言（不是 ImportError／fixture error）；實作後同指令 → GREEN。每個 RED 在不可變 test commit、GREEN 在後續 code commit，由 reviewer 依 Fleet Protocol 獨立確認。
**Regression:** `QT_QPA_PLATFORM=offscreen uv run --extra dev --extra research-ui python -m pytest tests/live/test_g3_*.py tests/live/test_qt_window.py tests/research/test_cli_lifecycle.py -q`；`uv run --extra dev ruff check src tests`；`uv run --extra dev mypy src`；`git diff --check`。保留 fake、headless、checkpoint、fixed-window。reviewer/lead 必須說明這個 PR 在舊 UI 下合併後不 broken。

## PR-B — operator Start-gated 流程與 SOP（依賴 PR-A）

**Goal:** 將 G3 改為 Ready（選相機不開）→ Running（Start 才 open＋即時預覽）→ Result（先 release、清空照片、文字結果）→ 標註成功後 Ready（保留選相機、照片和結果清空）。Cancel／關窗也關鏡頭；區分 no frame/no face/quality-rejected 與真正 unknown。
**Files:** `src/facecore/live/qt_window.py`、`src/facecore/research/cli.py`，必要時 `src/facecore/live/desktop.py`／`controller.py`；擴充 `tests/live/test_g3_start_gated_flow.py`、`tests/live/test_g3_commit_on_label.py`；更新 `docs/g3-local-test-sop.md`、`.github/workflows/ci.yml` qt-smoke job。
**Consumes:** PR-A 的每輪資料/clock 接口與 D0 取代後的 spec。
**Produces:** 選相機無 `open/read`；Start 開選定鏡頭、綁 5s；matched 提前或 timeout 終局一律先釋放；結果階段預覽 `QPixmap`/cache 清空；標註即時存檔且 reset；原選項 retained。
**Test-first evidence:** 新 case 分別從 G3 真入口證明「選相機後 `open/read==0`」「Start 後不同 frame 更新 UI」「match 提前／5s 到期才 terminal」「無幀／無臉／品質拒絕的結果與註冊未命中分別顯示」「結果／Cancel／關窗 release 且 pixmap 清空」「兩輪每次都需 Start，已標註 CSV、加密影像、trace 均可重開讀」「任一存檔失敗不前進」。修前 picker 選擇即開，且結果預覽不清；這是 RED 的預期失敗。真人鏡頭與臉一律不進測試。
**RED/GREEN:** `QT_QPA_PLATFORM=offscreen uv run --extra dev --extra research-ui python -m pytest tests/live/test_g3_start_gated_flow.py tests/live/test_g3_commit_on_label.py -q` → 修前針對新契約 RED、修後 GREEN；`QT_QPA_PLATFORM=offscreen uv run --extra dev --extra research-ui python -m pytest tests/live/test_g3_*.py tests/live/test_qt_window.py tests/research/test_cli_lifecycle.py -q` → regression GREEN。舊測試若保護自動待機語意，改寫成 Start-gated 之前先在獨立 test commit 證明 RED，不直接刪掉。
**Project verification:** `uv run --extra dev --extra research-ui python -m pytest tests/ -q`、`uv run --extra dev ruff check src tests`、`uv run --extra dev mypy src`、`git diff --check`；CI verify matrix＋qt-smoke matrix 實際執行新增整合測試並全綠，不能靠僅本機 pass（workflow 本身變更須以該 PR 的 CI runtime evidence 審查）。
**SOP:** 刪去「選相機後應預覽／不要按 Start」，明寫 Start 才開、結果關、標註後清空並待下一次 Start；已保存指示只在完整存檔後亮。spec §5 不做項目保持不變。PR merge＋landed CI 後由 operator 自行用真相機驗：選相機時燈不亮，按 Start 後亮且畫面有動，結果顯示前熄燈，標註後無照片、再按 Start 可開始下一輪；失敗以去識別文字回報。

## PR／責任邊界與交付

1. **Docs-only D0 PR**：`project-docs-maintain` check、spec 取代、必要 PROJECT-STATE pointer。docs 先落，但明說實作待交付；不提前更新 SOP 成已可使用。
2. **PR-A**：資料與時計接線；只在可獨立合併且不破壞現有 App 時單獨成 PR。若 D1 證實與新生命週期不可拆，與 PR-B 合成一個實作 PR並以此為唯一可合併單位。
3. **PR-B**：Start-gated 行為、SOP、CI qt-smoke。PR-A 先 merge，再從新 main 分支；不疊未 merge 分支。每 PR 依 lead 凍結 review_class、review/CI/landed watch 流程。
4. **Completion claim**：只在 offscreen 真入口測試、review、PR CI、landed exact-head CI 各自有證據後說「工程已交付」；真機辨識仍以 operator 實測為準，不能由 synthetic 結果替代。

**Highest risks:** 真入口測試偷繞 picker／工廠；每輪初始 session 尚 active 使影像無法 stage；裁切／preview 使用不同影格；錯的 clock 使 5 秒失真；終局相機仍被 background reader 持有；Qt 結果畫面保留人臉；CI 未執行新增 Qt tests。各風險已有對應 D1／PR-A／PR-B acceptance，不另加非目標治理。
