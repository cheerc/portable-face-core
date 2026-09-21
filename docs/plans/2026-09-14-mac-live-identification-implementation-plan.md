# Phase 2A Mac 動態辨識研究原型 Implementation Plan

**Source of truth:** [研究設計](../specs/2026-09-14-mac-live-identification-research-design.md)、[ADR 0008](../decisions/0008-mac-live-identification-research.md)，於 PR #39 合併（`fa16843e12b90fd8bfe7ded29146a0af43af7fae`）。Operator 2026-09-14 已批准寫 plan 並交 lead 正常 review/PR/merge。

**Goal:** 在 Mac 以固定單照 gallery 執行有界單人相機 session，能引導、標記、經同意加密保存並公平回放比較單幀／多幀策略。

**Architecture:** 相機與本機 UI 是薄 adapter；純 session engine 消費有序、已評分 observation。研究 recorder/evaluator 與正式 identity repository、learning confirmation 分離；第一版不呼叫 candidate/promotion。

**Tech stack:** 現有 Python、NumPy、Pillow、ONNX Runtime、pytest/ruff/mypy；GUI/capture 套件、直接加密寫入方式須先由 S1/S2 有界 spike 產生可重現證據，再經 docs-only 選型補丁凍結。

**授權狀態：plan-only。** 本計畫 merge 不授權執行 spike、安裝新依賴、啟動相機、錄製真人或實作產品；operator 明確實作 go 後依下列 gate 執行。尚未有 GUI/camera benchmark 結果；以下新 symbols／commands 都是擬新增契約，不冒充現有 API。

## 1. Global Constraints

- 固定 gallery：每人一張註冊照、open-set 1:N；gallery 至少兩人，單身份無 runner-up 不支援自動 matched。可由既有單照在隔離記憶體 gallery 建立；第一版不直接掛可寫正式 DB。
- 研究保持 offline；無 HTTP/server、mobile、雲端、考勤、authenticated、foundation training。未知模型／generation／policy 不相容即拒絕。
- 單人、有界 session；多臉或 continuity 不明清空窗口並要求重新開始。不用多幀一致性冒充 liveness。
- 第一版固定 policy profile、5秒窗口、相鄰取樣至少相隔200ms／至多26幀（含 t=0 與 t=5000 兩端點）、最新幀槽1；不要求實際硬體達到5fps，也不因慢而延長deadline。工程預算變更須版本化並回設計review。
- 敏感資料不進Git／一般log／plaintext temp；評估紀錄與影像分開同意、分開TTL（30/7天）、分開key，以免7天影像過期需連帶刪除30天紀錄。記錄同意不等於learning confirmation。
- 既有 detector/match/margin/governance defaults 不改。研究 match/margin 要求由使用者提供的versioned profile明列，不選產品預設；0.15不得由M5自動升格。
- Task的RED/GREEN為預期指令與失敗／成功形狀，不是已跑結果；test-first遵循fleet immutable RED→GREEN規則。

## 2. Evidence 與未證明假設

已知來源：`pipeline/decode.py` encoded-image入口、`pipeline/yunet.py` detector、`align.py`、`measure.py`、`quality.py`、`embed.py`、`policy/identify.py`、`contracts/result.py`；`eval/session.py` 為單照管線參考，不是live session engine。`eval/real_replay.py` 只為離線評估。`governance/corroboration.py` 的burst suppression不能以相鄰幀繞過。

M1/M5靜態結果及M3 13事件負向更新證據只支持其原始條件；不保證camera session效果、不代表M1 negative曾進M3、不支持時間老化趨勢。版本與限制見PROJECT-STATE。

未證明：Mac capture/UI 可否正確處理權限與Stop；ORT teardown是否可靠；新decoded-frame入口可否與encoded pipeline一致；AEAD recorder是否能無plaintext temp地完成崩潰恢復；多幀是否降低session誤接受且維持成功。分別由S1、T2、S2/T5、T8驗證；若缺證據，該路徑blocked，不猜結果。

### 2.1 已核對的既有介面與重用限制

- `pipeline/decode.py`：`DecodedImage(width: int, height: int, color_order: str, pixels: bytes)`；`decode_image(data: bytes) -> DecodedImage`。新adapter正規化後直接構造RGB raw bytes物件，不把ndarray冒充encoded bytes。
- `YuNetDetector(artifact_path: Path, expected_sha256: str, *, input_size: int | None = 640)`；`detect(decoded, *, score_threshold=0.9, nms_threshold=0.3, top_k=5000)`。detector/embedder一次初始化、多幀重用；目前CPU provider與固定640 resize不能假稱已達live性能。
- `align_crop(raw_rgb: bytes, width: int, height: int, face: DetectedFace)`目前contract v1為box crop/NEAREST，而非landmark similarity transform；T2保持一致，不順手換alignment。`sharpness_of(crop)`尺度綁112 crop；pose為proxy、landmark confidence固定1.0，不能以品質提示宣稱完整遮擋偵測。
- `identify(probe, repository, policy, *, gallery, model_version, quality=None, drift_status=None)`要求policy已明列match/review/margin；`frozen_v1()`的空threshold不能直接上線。ScoringContext把ResearchProfile的三門以`with_thresholds(match_threshold=..., review_threshold=..., margin_threshold=...)`轉換，並驗`review_threshold <= match_threshold`；metadata repository使用固定記憶體snapshot，不每幀讀寫加密DB。
- `storage/cipher.py`：`AeadCipher(dek: bytes).encrypt(plaintext: bytes, aad: bytes) -> EncryptedBlob`、`decrypt(blob, aad) -> bytes`；key為32 bytes。`EncryptedBlob.to_dict()`是既有hex表示；SQLite私有`_seal/_parse_blob`不是可供recorder呼叫的公共API。S2凍結research envelope serializer與AAD，測cross-session/kind/index替換必失敗；不複製私有SQLite格式假稱相容。
- `FileKeyProvider`是identity命名語義且預設`~/.facecore/keys`；research adapter必須顯式獨立key/master路徑，禁止落到正式預設或沿用正式環境key。每session、每類別分key且影像上限26幀（含 t=0 與 t=5000 兩端點）；重試不得重用nonce/ciphertext配對，刪除須處理所有重試產生的key。
- `pyproject.toml`釘Python 3.14、ORT 1.30.0、NumPy 2.5.3、Pillow 12.3.0；GUI相容性是S1待驗。現有CI為`python -m pytest tests/ -q`、`ruff check src tests`、`mypy src`；scale用`python scripts/run_500_scale.py`。

## 3. Documentation gate（所有實作前 blocking）

`docs-check: arch=N adr=N area=Y`：架構已由ADR0008批准方向，本plan細化area interfaces；本次修改README map、PROJECT-STATE、新設計status/pointer。搜尋範圍：`Mac`、`Phase 2A`、`implementation`、`recorder` 於CLAUDE、PROJECT-STATE、ADR0008、研究spec、README。

**D0（docs-only）**：lead 驗本plan/spec/ADR相容、相對連結、來源SHA與operator go。先跑 `project-docs-maintain`；記錄文件 gate 已過，再准S1/S2。若S1/S2改變選型或契約，先由作者更新spike manifest與plan、review/merge（D1），才准T1起的產品程式。不能把docs-check僅放在最後驗收。

舊1B plan是歷史工作令，不需重開P0。每PR記錄自己的live base/head；不能把本plan source SHA當成永久必須checkout的HEAD。

## 4. File／Interface／Dependency Map

擬新增元件（下面的簽名是本plan定義的新介面，不是現有symbol）：

| 路徑 | 職責 |
| --- | --- |
| `src/facecore/live/__init__.py`, `contracts.py` | immutable frame/profile/observation/session schema |
| `src/facecore/live/session.py` | 純有界時間一致性及最佳品質單幀判定 |
| `src/facecore/live/frame_pipeline.py` | decoded camera frame→既有quality/scoring，無truth |
| `src/facecore/live/capture.py` | CaptureSource protocol、latest-only pump |
| `src/facecore/live/desktop.py`, `controller.py` | 本機視窗、UI/capture/scoring生命週期與freeze |
| `src/facecore/research/__init__.py`, `records.py` | session/consent/label envelope |
| `src/facecore/research/recorder.py`, `keys.py` | 加密bundle、per-session資料類別key、TTL/delete/recovery |
| `src/facecore/research/replay.py`, `report.py` | sealed資料回放、分母、holdout與策略對比 |
| `src/facecore/research/cli.py` | 隔離研究入口；不更改既有CLI命令 |
| `tests/live/{test_contracts,test_session,test_frame_pipeline,test_capture,test_desktop}.py` | synthetic/fake元件測試 |
| `tests/research/{test_records,test_recorder,test_replay,test_report,test_cli}.py` | 存取失敗、truth isolation、端到端 |
| `docs/research/2026-09-14-mac-live-spike-manifest.md` | S1/S2 evidence與選型，D1前新增 |
| `docs/research/mac-live-runbook.md` | 可執行操作、相機/同意/刪除smoke，T7新增 |

**新契約：**

- `FramePacket(sequence: int, captured_ns: int, rgb: ndarray, orientation: int, mirrored: bool)`：正整數sequence、monotonic ns、uint8 H×W×3、擁有buffer不借用camera mutable view。
- `ResearchProfile`：schema/version/hash、timeout_ms=5000、sample_interval_ms=200、max_frames=26、queue_limit=1、required_support=3、min_support_interval_ms=200、match_threshold、review_threshold、margin_threshold、detector/quality policy版本、continuity_max_center_delta_ratio。thresholds與continuity界線均須明列finite且合法；缺值拒絕啟動。跨欄位必要條件 `max_frames >= ceil(timeout_ms / sample_interval_ms) + 1`（5000／200 即 26；第一幀 t=0，後續至少相隔 200ms，第 26 幀名目 t=5000ms）；不滿足則完整 deadline window 不可達，必須 fail closed／拒絕開始（production 實作另待 plan＋operator go）。continuity界線由S1非人臉移動標靶測試提出並D1凍結，不宣稱能保證同一真人。
- `FrameObservation`：sequence、captured_ns、processed_ns、quality_pass/reasons、face_count、face_box、identity_scores（opaque IDs→finite scores）、quality_rank、model_generation、gallery_digest；沒有ground_truth/display_name。
- `SessionResult`：session_id、schema_version、status（matched/review/unknown/invalid_input/timeout/cancelled/error）、opaque identity（僅matched）、reason_codes、elapsed_ms、sampled/usable/rejected/dropped、support_sequences、profile/model/gallery版本。session狀態不擴改既有單幀IdentificationResult enum。
- `SessionEngine(profile, gallery_digest, model_generation)`；`start(session_id: str, now_ns: int) -> None`；`observe(observation: FrameObservation) -> SessionResult | None`；`finish(now_ns: int, reason: str) -> SessionResult`。每次start對應唯一immutable終局；terminal後observe不可新增支持。
- `score_frame(frame: FramePacket, context: ScoringContext) -> FrameObservation`：context持固定gallery/model/policy；不能讀labels/recorder/UI。
- `CaptureSource.open(device_id: str) -> None`、`read() -> FramePacket | None`、`close() -> None`；FakeCapture提供deterministic事件流；真實套件僅藏在此adapter內。
- `ResearchRecorder.begin(session_id, consent: ConsentRecord) -> None`；`append_frame(frame: FramePacket) -> None`；`commit(result: SessionResult) -> None`；`abort(reason: str) -> None`；`delete(session_id: str) -> None`；`purge_expired(now_utc) -> None`。影像consent缺席時append_frame拒絕，不靜默保存。
- `replay_session(bundle_id: str, profile: ResearchProfile, context: ScoringContext) -> SessionResult`；`summarize(results, labels) -> ResearchReport`；標籤只在summarize使用。

**資料隔離：** `ScoringContext`只持read-only研究gallery（每輪從原始單照重建、digest凍結）；不接受生命週期mutation服務。研究records與影像不同key namespace，不重用正式identity DEK；無正式DB migration。研究schema不相容時fail-closed，不自動升級或丟棄內容；第一版無匯出／備份／同步。rollback指research pipeline/profile版本重播，不改正式模板。

**依賴：** D0→S1/S2→D1→T1；T1→T2/T3/T5；T2+T3→T4；T2+T3+T5→T6；T4+T5+T6→T7→T8。每Task独立PR，不能因方便把兩個獨立行為合併。

## 5. 有界 Spike（未執行）

### S1：Mac capture/UI／runtime可行性（上限1個工作日）

Owner：lead指派implementer；reviewer獨立核evidence。候選優先測OpenCV capture＋薄desktop GUI；若wheel/headless/GUI thread不相容，評估AVFoundation adapter路線；不因未測就指定可行。

問題：權限拒絕/允許、camera device枚舉、orientation/鏡像/色序、preview主執行緒、5秒取樣上限、斷線/Stop/UI close、ORT teardown、macOS arm64 packaging/license。

僅測空背景／非人臉標靶；不下載模型或保存真人。產物：`experiments/mac_live_capture_probe.py`（不列正常app入口）、上述spike manifest，附exact環境、命令、exit code、裝置釋放證據、依賴license與可重現camera-free模式。captured pixels不進報告或Git。

停止：超時、native crash、Stop無法釋放、依賴license未清楚或必須server/mobile才可工作。報blocked與最小替代路線，不擴scope。後續沿同spike artifact/hash，不重新做無紀錄實驗；真機程式不可作無tests的產品拷貝。

### S2：研究AEAD recorder／key及刪除可行性（上限1個工作日）

Owner：implementer；使用synthetic pixel payload。核既有cipher/KeyProvider能否重用純primitive而不把研究session偽裝為identity；AEAD AAD需綁schema/session/data-kind/frame-index，禁止改加密演算法。

驗證：encrypt-before-write、key分離、atomic manifest/commit、partial encrypted blob恢复、30/7天分離到期、刪除時先禁止讀取再銷key/清索引的可重入流程、key unavailable/磁碟滿、已同意negative保存但不學習。

產物：`experiments/mac_live_recorder_probe.py`、同spike manifest的key layout/檔案layout/AAD/test命令與crash matrix。停止：明文temp、跨重啟刪除失敗、key遺失後仍能讀、不得不改正式DB或超時。

### D1：spike結論凍結（docs-only，產品實作前blocking）

作者更新manifest與本plan：具體dependency版本/GUI選型、continuity界線、key adapter signatures及layout、fake與真機命令。reviewer確認來源；lead正常merge。若沒有這些結果，T1–T8不得開工。這是有界研究出口，不是「TBD後再隨意決定」。

## 6. 實作 Tasks

以下命令中的測試檔為待新增，RED在寫tests後、寫implementation前執行。`uv run pytest <file> -q`先因缺介面或指定assertion失敗；GREEN為同指令全部通過。先用`python -m pip install -e '.[dev]'`安裝現有dev依賴，或以現有uv環境執行。每task專案驗證對齊已核CI：`uv run ruff check src tests`、`uv run mypy src`、`uv run python -m pytest tests/ -q`、`git diff --check`。新GUI dependency僅D1批准後才能安裝；新增研究extra必須在T7 CI有對應安裝／import smoke，非macOS平台可跑fake tests，不得為缺native library而整套跳過。

### T1：研究契約與版本化 profile

Files：Create `src/facecore/live/__init__.py`, `contracts.py`, `src/facecore/research/__init__.py`, `records.py`；Test `tests/live/test_contracts.py`, `tests/research/test_records.py`（新增test目錄的`__init__.py`）。Consumes D1；Produces §4 immutable contracts。

Acceptance：拒絕NaN/Inf、負時間、未知schema、缺threshold/profile、錯RGB shape/dtype；JSON往返保留版本；result非matched無identity；consent分record/image兩權限與期限；label不能成為FrameObservation欄位。

RED：`uv run pytest tests/live/test_contracts.py tests/research/test_records.py -q` → missing contracts或非法profile被接受。Minimal：dataclasses＋明確validate/serialization，無camera與crypto side effects。GREEN：同命令通過。

### T2：decoded-frame單幀管線與固定 gallery

Files：Create `src/facecore/live/frame_pipeline.py`；Test `tests/live/test_frame_pipeline.py`。Consumes FramePacket/profile與既有pipeline；Produces score_frame/ScoringContext。

Acceptance：encoded fixture與同像素frame在orientation/color normalization後等價；mirror僅preview不改推論；single-face/品質拒絕不embed；model mismatch拒絕；scores含runner-up並用原有policy規則；quality_rank是明確品質指標，不讀identity score。以通過品質後sharpness排序、相同值以sequence較早者優先，數值定義沿既有measure。

從corpus manifest每人單照建立研究gallery，拒絕重複identity/缺檔/不合格註冊；完成前不能start，不留部分gallery。本task不接正式DB。

RED：`uv run pytest tests/live/test_frame_pipeline.py -q` → 色序/鏡像不一致、multi-face仍embed或gallery被改寫。Minimal：純frame adapter呼叫既有detect/quality/align/embed/identify，提供injected test doubles。GREEN：同命令通過；增加既有 `tests/pipeline`、`tests/conformance` 回歸，不以真人照片入repo。

### T3：有界 session 與兩策略

Files：Create `src/facecore/live/session.py`；Test `tests/live/test_session.py`。Consumes FrameObservation；Produces SessionEngine＋品質最佳單幀baseline。

精確規則：deadline=start+5秒；相鄰有效支持至少200ms、同identity至少3幀且各過match/margin；低品質/零臉清支持窗口，multi-face/continuity不明終止為invalid_input需重啟；新合格幀支持其他identity清支持再重新累積。NaN、generation/digest變更、時間倒退/duplicate sequence終止error；處理時已超deadline不得追認舊幀成功。合法但未過match/margin的幀清支持，禁止分散尖峰累積。terminal呼叫冪等回同結果。

live session首次達條件可提前matched；公平benchmark必須另外以同5秒封存流回放所有策略，不能把已提前停止的短流冒充完整5秒對照。baseline只在封存窗口結束選品質最高合格單幀；None margin不得matched。到期有合格但不足接受→timeout，附best-frame band作diagnostic；完全無合格→invalid_input；cancel與error分開。

RED：`uv run pytest tests/live/test_session.py -q` → any-frame-wins、跨人累積、deadline延长、terminal二次決策。Minimal：純狀態機＋injected monotonic clock。GREEN：同命令通過；同inputs/profile重播逐event與結果一致。

### T4：capture pump／controller（不接真實資料保存）

Files：Create `src/facecore/live/capture.py`, `controller.py`；Test `tests/live/test_capture.py`。Consumes T2/T3、D1已選capture backend；Produces bounded camera controller。

Acceptance：latest-slot1、相鄰取樣至少相隔200ms、最多26幀（含 t=0 與 t=5000 兩端點）、drop counters；UI收到舊session推論結果必丟棄；Stop/斷線/worker failure/close釋放camera及join worker，不能fire-and-forget；deadline用controller clock而非模型完成時刻。暫不公開產品入口。

RED：`uv run pytest tests/live/test_capture.py -q` → fast producer無界queue、stale結果污染新session、close遺留worker。Minimal：CaptureSource/FakeCapture＋有限worker ownership＋one-slot queue。GREEN：同命令通過；S1命令在實作adapter下重跑非人臉smoke，timeout與native crash不得當成功。

### T5：同意、加密、TTL、可恢復刪除

Files：Create `src/facecore/research/recorder.py`, `keys.py`；Test `tests/research/test_recorder.py`。Consumes Records與D1 crypto layout；Produces ResearchRecorder。

Acceptance：record/image分開consent和key；image<=26張/5秒（含 t=0 與 t=5000 兩端點）；未同意/撤回/multi-face清未commit影像；可保存已同意negative但不接learning。先加密後write/atomic manifest；partial檔在恢復先reconcile，禁止讀取未commit bundle。記錄30天與影像7天独立expiry；purge在啟動及寫前，wall-clock倒退標時鐘錯誤並停止新增保留，不延長既有expire_at。delete tombstone先封讀，再刪對應keys/blobs/labels/索引與衍生資料；失敗回error並重啟重試，不宣稱已刪。

RED：`uv run pytest tests/research/test_recorder.py -q` → 無同意落盤、7天expiry仍能解密frame、刪除後重啟可讀、tamper被接受。Minimal：復用審核過AEAD primitive與隔離namespace，injected clock/key provider/filesystem failures。GREEN：同命令通過；每個write/rename/key/delete邊界注入failure驗recovery。測试只用synthetic，不落真實圖。

### T6：回放、ground truth隔離與報告

Files：Create `src/facecore/research/replay.py`, `report.py`；Test `tests/research/test_replay.py`, `test_report.py`。Consumes committed bundles/scorer/engine；Produces replay_session、summarize。

Acceptance：同bundle/profile/gallery/model→相同結果；改label不改inference；檔案缺失/過期/刪除/manifest tamper拒絕回放，分開標error而非unknown。沒有frame的record僅能分析既有分數，不能宣稱換模型重跑。完整窗口/提前截止bundle明列；新模型需重新由同註冊照建立generation，不混embedding。

報告：attempted、labeled、eligible、successful、correct identification、wrong enrolled identity、unknown false accept、timeout/invalid/error/cancel逐項數；不同profile同session配對，省略的session及原因必列；成功latency及全嘗試含censoring分開。development/holdout依visit/session分組；holdout被調參使用即重標development，不把幀随机切分。無具代表性樣本不宣稱跨身份或500人有效。

RED：`uv run pytest tests/research/test_replay.py tests/research/test_report.py -q` → label leak、遺漏error分母、把early短流当完整窗口。Minimal：decrypt→replay與labels-evaluate兩階段；分母明列。GREEN：同命令通過。

### T7：本機視窗與研究CLI端到端入口

Files：Create `src/facecore/live/desktop.py`, `src/facecore/research/cli.py`, `docs/research/mac-live-runbook.md`；Modify `pyproject.toml`（僅D1核准optional research dependencies）、`README.md`、`docs/PROJECT-STATE.md`；Test `tests/live/test_desktop.py`, `tests/research/test_cli.py`。

新命令契約（本task完成前不可執行）：`python -m facecore.research.cli live --corpus <external-manifest> --models <external-model-dir> --profile <external-profile-json> --store <external-research-dir> --device <id>`；`replay --store <dir> --session <id> --profile <json> --corpus <manifest> --models <dir>`；`delete --store <dir> --session <id>`。init/consent在UI首次顯示；缺profile/key不得預設開錄，store須拒絕repo內路徑與symlink逃逸，不宣稱可自動辨識所有雲端同步目錄（runbook要求非同步位置）。

UI：開始／取消／5秒倒數、相機權限、臉框／品質提示、研究非認證標籤、matched才顯示身份、終局後操作者標記真實身份/unknown/unlabeled。主動勾選record/image兩同意；recording indicator從啟用起持續顯示。預設live可提前結束；另明確「固定5秒比較」研究模式，inference終局鎖定但capture只在已同意recording下到原deadline，供公平回放；取消/多人/錯誤立即停止，不能為湊窗口繼續錄。

RED：`uv run pytest tests/live/test_desktop.py tests/research/test_cli.py -q` → Start繞過同意/key、uncertain露名字、UI close未釋放、輸入truth進scorer。Minimal：真正CLI→controller→fake capture→真engine→test recorder端到端，不能只測helper。GREEN：同命令通過；既有CLI回歸綠。若GUI不能headless測，測controller與事件binding，另記真機evidence，不能省略。

### T8：研究啟用驗收與文件收尾

Files：Modify `docs/research/mac-live-runbook.md`, `docs/PROJECT-STATE.md`；Test擴充 `tests/research/test_cli.py` lifecycle與故障矩陣。

Acceptance：T1–T7所有RED/GREEN與CI receipts在位；S1真機無人臉camera smoke在最終app版本重跑；permission拒絕、device斷線、cancel、Stop、window close、restart/purge/delete、tamper各有command/操作/退出狀態。

真人smoke僅operator另確認參與者同意後：既有gallery一人＋一位未註冊者，各1次session，記UI與終局對應、錄製開關、保存/回放/刪除；沒有參與者時標`hardware-ready; participant-smoke blocked`，不索取新增照片或假造完成。不以必須認對當工具成功標準；錯認如實留下評估分類。

RED：`uv run pytest tests/research/test_cli.py -q` → failure injection後遺留可讀bundle/worker或report漏failed attempt。GREEN：同命令通過，加完整專案驗證與runbook smoke。最終可宣稱工具就緒；不可宣稱演算法校準完成、實際誤認率0或準備正式部署。

## 7. PR 邊界與完成定義

D0本plan PR；S1、S2各一spike PR；D1單獨docs選型manifest PR；T1–T8各自PR，後者只在dependencies merge後建立live base。無UI的foundation可独立merge，只需測試契約成立；recorder不得靠UI才保證同意。

每PR：task source/spec section、exact base/head、RED/GREEN與CI、docs-check、non-goals、known limits、typed reviewer receipt。lead正常merge後核tree；plan/spec實質變更回作者codex修檔。reviewer對隱私/生命周期/權限風險可要求dual review，不因docs或prototype降低門檻。

本plan完成＝書面scope/interface/dependency/測試契約經review及merge。產品實作完成＝T8工程與真機證據，受真人同意gate限制。研究有效性＝後續sealed session比較，另行報告。三者不能互相替代。

（2026-09-15 修訂，decision `d-20260915071553820286-1`）上段第二項「產品實作完成」進一步界定為 **Phase 2A**＝T8 工程 ＋ 前處理正確性與不變性 ＋ §9 中團隊可自足的項目（UI 輸出對上、保存開關、真機故障路徑、一位已註冊參與者 smoke）。第三項「研究有效性」界定為 **Phase 2B**＝§5 對照臂、§8 分母、holdout 切分與未註冊參與者 session。**2A 不設準確率目標。** 三者仍不能互相替代。
