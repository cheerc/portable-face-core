# D0：Mac Demo 修復基準（2026-09-29）

本檔是 D0 的可重現基準，凍結 Mac Demo 修復（D1–D4）所使用的版本、環境、gallery 與既有結果。
**它不宣稱任何真機辨識已成功，也不宣稱 D1–D4 已完成。** 所有 D0 事實皆為 2026-09-29 在
`b92a2762a4f33a6f1fdd209bc70ad9b3e6b96c9e` 本機親查；命令與輸出記於本文各節。

## 1. 凍結基準

| 項目 | 值 | 取得方式 |
| --- | --- | --- |
| repo | `cheerc/portable-face-core`（`origin` = `https://github.com/cheerc/portable-face-core.git`） | `git remote -v` |
| 凍結 HEAD | `b92a2762a4f33a6f1fdd209bc70ad9b3e6b96c9e`（「feat(g3): Start-gated Ready/Running/Result lifecycle + SOP + qt-smoke (PR-B) (#115)」） | `git rev-parse HEAD` |
| 承接時工作樹 | 乾淨（`git status --short` 無輸出） | `git status --short` |
| 修復分支 | `demo-d0-baseline`（自 `b92a276` 起，base = `origin/main`） | `git log --oneline` |

修復期間若 HEAD 前進，**以實際 HEAD 為準並在回報中記錄差異**；不得靜默沿用本檔數字。
本檔所有行號引用都綁在 `b92a276`，換 HEAD 後需重新核對。

## 2. 環境

| 項目 | 值 | 取得方式 |
| --- | --- | --- |
| 平台 | macOS（Darwin 25.6.0），arm64 | `uname`／本機 |
| `uv` | 0.10.7 | `uv --version` |
| Python | CPython 3.14.7（`pyproject.toml` 要求 `==3.14.*`） | `uv run python -V` |
| onnxruntime | 1.30.0（`pyproject.toml` 釘選） | `uv run python -c "import onnxruntime"` |
| 測試依賴 | pytest 9.1.1／ruff 0.16.7／mypy 2.3.1（`dev` extra） | `uv sync --all-extras` |
| UI／真機擷取 | pyside6 6.11.2、opencv-python-headless 5.0.0.93（`research-ui` extra） | `uv sync --all-extras` |

**注意：系統 `python3` 沒有 onnxruntime，也沒有安裝本專案。** 所有命令必須經 `uv run` 執行，
不可直接用 `python3 -m facecore...`，否則會得到 `ModuleNotFoundError`。

## 3. 模型與 checksum

| 模型 | 檔案 | SHA-256 | repo 內釘選 |
| --- | --- | --- | --- |
| YuNet 偵測 | `~/facecore-models/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | `src/facecore/research/cli.py:71-73`（`TRUE_DETECTOR_SHA256`）一致 |
| SFace embedding | `~/facecore-models/face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | 由 `ModelManifest.sface_2021dec_fp32()`（`src/facecore/contracts/manifest.py`）描述 |

`shasum -a 256 ~/facecore-models/*` 取得上表。YuNet 的實際檔案 checksum 與 repo 釘選值相同，
`YuNetDetector` 會在載入時驗證；SFace 走 manifest 契約。兩者皆為既有權重，**D1–D4 不換模型**。

## 4. 本機設定（不進 Git）

設定檔：`~/Downloads/face_sample/_facecore/g3-local.json`（2026-09-24 commander 建立）

```json
{
  "enrollment_dir": "~/Downloads/face_sample/註冊組",
  "models_dir": "~/facecore-models",
  "store_dir": "~/Downloads/face_sample/_facecore/store",
  "key_dir": "~/Downloads/face_sample/_facecore/research_keys"
}
```

store 與 key 的**實際位置由 launcher 決定**：store = `~/Downloads/face_sample/_facecore/store`，
key = `~/Downloads/face_sample/_facecore/research_keys`。

**本機運行資料不在 repo root**：`~/portable-face-core/_facecore/` 與 worktree 內都不存在。
`_facecore` 在 `~/Downloads/face_sample/` 之下（`find` 實查），依賴 `g3-local.json` 指向而非相對路徑。
這一點與 plan／diagnosis 寫的「`_facecore/...`」相對表述不同，執行時以設定檔的絕對路徑為準。

## 5. Gallery 基準

- 註冊組：`~/Downloads/face_sample/註冊組`，**23 張**（`enroll-01` … `enroll-23`）
- 身分識別：**檔名 stem**（`enroll-23.jpg` → `enroll-23`），非人名、非資料夾遞迴
- 建構函式：`src/facecore/live/frame_pipeline.py:136` `build_gallery_from_folder`（既有函式，直接重用）
- 接受格式：`{.jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp}`（`GALLERY_IMAGE_SUFFIXES`）；每張須恰好一張臉，否則點名報錯

以**真模型在 `b92a276` 重算**（`build_gallery_from_folder` + `YuNetDetector` + `Embedder`）：

| 欄位 | 值 |
| --- | --- |
| identities | 23 |
| model_version | `face_recognition_sface_2021dec` |
| generation | `gen-1` |
| **gallery digest** | `e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f` |

**此 digest 與既有 29 輪 CSV 每輪的 `gallery_digest` 欄完全一致（29/29）**，
證明 9/24–9/29 期間模型檔與註冊組未漂移，既有結果可作為修復基準。

digest 演算法（`src/facecore/live/frame_pipeline.py:77` `compute_digest`）：對 **sorted 身分身 + 各
embedding 位元組**做 SHA-256，**不是**對檔名或檔案位元組。換模型或重算 gallery 都會改變此值。

## 6. Profile 與門檻（凍結，D1–D4 不得改）

`profiles/g3-v1.json`（`b92a276`）：

| 欄位 | 值 |
| --- | --- |
| `profile_version` | `g3-v1` |
| `match_threshold` | **0.363** |
| `margin_threshold` | **0.10** |
| `review_threshold` | 0.3 |
| `required_support` | 3（多幀三幀確認） |
| `timeout_ms` | 5000 |
| `sample_interval_ms` | 200 |
| `min_support_interval_ms` | 200 |
| `max_frames` | 26 |
| `queue_limit` | 1 |
| `quality_policy_version` | 1 |
| `detector_version` | `yunet` |
| `continuity_max_center_delta_ratio` | 0.5 |

既有 29 輪的 `profile_version` 皆為 `g3-v1`、`model_generation` 皆為 `gen-1`。

## 7. 既有結果摘要（29 輪，2026-09-24／09-29）

來源：`~/Downloads/face_sample/_facecore/store/results.csv`（唯讀，**未覆寫、未刪改**）

| 項目 | 值 |
| --- | --- |
| 總輪次 | 29 |
| 結果分布 | `invalid_input` 23、`timeout` 6、**`matched` 0** |
| 有 top1 的輪次 | 11，**全部**為 `enroll-23`（score 0.6338–0.6778） |
| `label_kind` | `uncertain` 26、`unenrolled` 3 |
| profile / generation | `g3-v1` / `gen-1`（29/29 一致） |

`invalid_input` 的 23 輪**全部** `frames_sampled=1`（分佈 `{1: 23}`，無任何一輪採到 2 張以上），
耗時 5264.2–34558.8 ms（下界為 `att-g3-20260924-192646-r16`）。
**全數只採到 1 張是更強的診斷訊號**：不是「多數輪次」只拿到一張，而是每一次都在首幀之後就耗盡了
5 秒窗口，支持 D1「首幀延遲吃掉整個辨識窗口」的假設。

`zero_usable_frames_collected` + `deadline_exceeded` 的失敗模式在
`src/facecore/live/session.py:504` 產生（無 usable observation 時 finish 為 `invalid_input`）。
6 輪 `timeout` 有 5–13 張樣本，耗時 5006.9–5221.5 ms。

**這些數字不能解讀為準確率。** `label_kind=uncertain` 只代表 operator 當時按了「錯誤」，
**不是系統已知的正確身分**；`unenrolled` 亦為按鍵標註。因此：

- **0/29 matched 不等於 SFace 準確率 0%**，也不等於辨識失敗。
- 11 輪的 top1 全為 `enroll-23` 可能是正確辨識、也可能是誤認，**現有標註無法區分**。
- 此 29 輪為**工程診斷樣本**，非有效準確率樣本（詳見 `docs/PROJECT-STATE.md`）。

### 診斷報告指出的現場失敗（唯讀定位，D0 不修）

| 位置 | 事實 |
| --- | --- |
| `src/facecore/live/qt_window.py:549` | `round_start_ns = self._clock_ns()` 在 `self.desktop.on_start(...)`（:550）**之前**求值。相機 open 發生在 `on_start` 內部（`controller.py:158`），故開相機耗時被算進 5 秒窗口。 |
| `src/facecore/live/qt_window.py:546-548` | 該處註解聲稱 clock 在「source open 之後」讀取，**與 :549 的實際求值順序不符**。PR-A 宣稱的 re-anchor 與程式實況有落差，屬 D1 釐清範圍。 |
| `src/facecore/live/controller.py:158-160` | `start_session` 先 `self._source.open(device_id)`，再 `self._engine.start(session_id, now_ns)`；engine 使用呼叫者提供的**開相機前**時間起算。 |

以上僅定位引用，**本 PR 不修改任何程式碼**（D1 範圍）。

## 8. 固定啟動方式（真機，一次一版）

依 `scripts/g3-local-test-app.command` 既有行為，落定如下：

1. **程式碼**：`~/portable-face-core`（canonical checkout）。D0 分支 `demo-d0-baseline` 僅承載文件變更；
   真機驗收前需由 Lead 決定在何處驗收本分支，**不得默默 `git pull` 切換版本**。
2. **設定**：`~/Downloads/face_sample/_facecore/g3-local.json`（已備妥，見 §4）。
3. **環境**：`uv sync --extra research-ui`（launcher 自動執行）。
4. **啟動**：Finder 雙擊 `~/portable-face-core/scripts/g3-local-test-app.command`；
   Terminal 會開啟並 `exec` 該 App（Qt 視窗）。
5. **操作**：依 `docs/g3-local-test-sop.md`：選相機（不開鏡頭）→ 勾兩個 consent → 按 **Start** → 辨識 5 秒 → 按「正確／錯誤」。
6. **結果**：`~/Downloads/face_sample/_facecore/store/results.csv`（逐輪摘要）＋同層加密診斷。

**可重現性警告**（D4 前置 PR 後已解除）：launcher 原會在啟動時 `git pull --ff-only origin main`，
版本會被更新。現已改為**只顯示**目前 commit 與落後狀態、**不修改 repo**（decision `-14`），
真機驗收時把 Terminal 印出的 commit 記入記錄即可。詳見 §11 待確認清單第 5、12 項。

操作細節（相機權限、卡住時對照表、資料保存 30 天規則）以
[`g3-local-test-sop.md`](g3-local-test-sop.md) 為準，本檔不重複。

## 9. 不開相機的重現方式（camera-free）

以 repo **既有** fake 入口，無自造 flag。`--store` 與 `--key-dir` **皆為必填**（缺一即 exit 2），
下列命令已逐字實測可執行（2026-09-29，`b92a276`）。`$(mktemp -d)` 建立暫存目錄，
確保 copy-paste 後直接可跑，且**不觸碰既有 29 輪結果**：

```sh
cd ~/portable-face-core   # 或 D0 worktree
D0_TMP="$(mktemp -d)"
QT_QPA_PLATFORM=offscreen uv run --extra research-ui \
  python -m facecore.research.cli live \
    --device fake \
    --profile profiles/g3-v1.json \
    --ui fake \
    --session d0-fake-smoke \
    --store "$D0_TMP/store" \
    --key-dir "$D0_TMP/keys" \
    --record-consent --image-consent
rm -rf "$D0_TMP"
```

實測輸出（2026-09-29，暫存 store／key，未觸碰既有結果；exit 0）：

```
{"elapsed_ms": 600.0, "gallery_digest": "cli-fake-gallery", "generation": "cli-fake-gen-1",
 "reason_codes": ["supported_3_frames"], "session_id": "d0-fake-smoke",
 "status": "matched", "window": "early-stop"}
```

| 項目 | 說明 |
| --- | --- |
| 入口 | `src/facecore/research/cli.py` `live` 子命令（`--device fake` 見該檔 :17-18、:732-750） |
| 假 source | `src/facecore/live/capture.py:97` `FakeCapture`（決定性合成影格） |
| 假 scorer | `_fake_scorer`（固定分數，不載真模型）→ 故 digest 為 `cli-fake-gallery`，**不可與 §5 真 digest 比較** |
| 必填旗標 A | **`--store` 必填**。省略（且無 `--config`）時 `research live` 拒絕執行並 `exit 2`（`cli.py:2051-2057`）。必須指向可寫入的目錄。 |
| 必填旗標 B | **`--record-consent --image-consent` 兩者皆必填**，缺一即拒絕執行（實測） |
| 隔離 | `--store` / `--key-dir` 必須指向**暫存目錄**，**絕不可**指向 `~/Downloads/face_sample/_facecore/store`，否則會把本輪混入既有 29 輪 |
| 無相機 | `--device fake` 走合成 pump，不開 AVFoundation，適合 CI 與離線重現 |

**D1 的計時失敗重現**（沿用 diagnosis 的三幀思路，但用 repo 既有測試入口，不自造腳本）：
`tests/live/test_g3_start_gated_flow.py` 已以 `FakeCapture` + 假時鐘 + offscreen Qt 覆蓋
Start-gated 流程與 `zero_usable_frames_collected` 失敗模式
（:325-345 `test_low_quality_round_does_not_finish_before_deadline`）。D1 應擴充此類測試，
而非另建重現機制。

## 10. Baseline 驗證（`b92a276`，全綠）

依 `.github/workflows/ci.yml` 的既有步驟（非猜測）：

```sh
uv sync --all-extras
QT_QPA_PLATFORM=offscreen uv run --extra dev --extra research-ui python -m pytest tests/ -q
# → 867 passed, 8 skipped in 323.91s
```

目標子集：`pytest tests/live/test_g3_start_gated_flow.py -q` → 10 passed。

D0 僅變更文件，故 regression 以上述 repo baseline 為準。

## 11. 尚待真機確認的問題清單（本 PR 只列，不解）

以下**全部未經真機確認**，任一項都不可由本檔或既有 29 輪推定：

| # | 問題 | 為何還沒答案 | 預計釐清點 |
| --- | --- | --- | --- |
| 1 | 首幀延遲中 **open 與 read 各占多少** | 現有 trace 無 open-start／open-done／first-frame 分段時間量測。**D4 真機第一輪（2026-09-30，32 列）仍拿不到**：`elapsed_ms` 由 5.5s 到 260.8s（中位數 15.2s），含 operator 調整站位與中途放手的時間，**不能當辨識速度**；計時歸因要看 `TimingMarks`，該輪 operator 未量到 | D1（加分段量測）＋ D4 重跑時請一併量 |
| 2 | **UI 卡頓**實際程度 | Qt timer callback（`qt_window.py:276` 接線、`:598` 同步 `run_until_terminal`）走在主執行緒。**D4 真機第一輪仍待驗**：該輪 32 列沒有任何可歸因的 UI 時間量測，只有總 `elapsed_ms`（含人為時間，見第 1 項） | D2（worker 化後實測 preview FPS／最大停頓）＋ D4 重跑時請一併量 |
| 3 | **曝光拒絕是場景問題還是量測／門檻問題** | 9/24 r4 trace 多張 `quality_exposure`，尚無畫面條件對照。**D4 真機第一輪有第一批真機樣本，但只涵蓋一個原因分支**（實測見下方說明）：16 列 `invalid_input` 中僅 **5 列**是「拍了影格但全部被拒」（2 列 `all_frames_rejected_quality`、2 列 `all_frames_rejected_mixed_causes`、1 列 `all_frames_rejected_no_face`），其餘 11 列是 `no_frames_captured`（2+2+1+11 = 16 ✓）。**畫面條件（正常光／背光對照）這一輪 operator 未做，單人單場景無法斷言是場景問題還是門檻問題** | D1 量測後由 D3 呈現區分；D4 重跑時請做光照對照 |
| 4 | 現有模型在**代表性人群的準確率** | 29 輪為單人工程診斷樣本，標註不可信；13 辨識組／30 non-target 尚未以現版本重跑 | D5（D4 之後） |
| 5 | **launcher auto-`git pull`** 對驗收可重現性的影響 | D4 要求固定版本驗收，但 `.command` 啟動即快轉 main | **D4 前置 PR 已解**（decision `-14`）：launcher 改為顯示目前 commit 與落後狀態、不再 pull。仍需真機確認顯示正確 |
| 6 | 分段時間量測（open 開始／完成、首幀、推論開始／完成、結果完成） | 尚未實作 | D1 |
| 7 | 首幀 open／read 分攤的**真機**數值 | 合成重現無法回答真機相機行為 | D1 量測＋D4 實測 |
| 8 | `ruff check .`（全 repo）有 1 個 E501 | `experiments/mac_live_capture_probe.py:521`；CI 只跑 `ruff check src tests` 故綠。非本 PR 引入，**本次不併入 D0 修正**（保持 source 零變更的可審計形態） | 之後的 lint 清理 |
| 9 | `close()` 的 **join 逾時路徑**真機行為 | D2 讓真機 GUI 有 worker 停在原生 read 內；read 若活得比 5s join 預算久，`close()` 會帶著**仍開啟的相機**返回（刻意的取捨，見 `controller.close()` docstring 與 `_stop_and_release()` 既有判定）。合成環境只能證明「不釋放、不崩潰、有界」，**證明不了真機 AVFoundation 的 read 是否可能逾時 5s、逾時後相機是否確實釋放、使用者是否看得見燈號殘留** | D4 實測 |
| 10 | **UI 與下游報表統計語意不一致（known divergence）** | `report.py:147`／`:175` 與 `analysis.py:627` 以 `status==timeout` 歸類，把 `insufficient_evidence` 計入 timeouts；UI 以 reason codes 區分「已看見人臉，但多幀確認未成立」與一般超時（找不到人員）。commander 裁定新舊分母刻意並存、不重新詮釋既有報表分類，D3 只保證新紀錄帶可區分的 reason code、模式與版本供日後分析分流 | D3（紀錄保留 reason code，分類維持現狀） |
| 11 | `tests/cli/test_derive_int8bq.py:63` 寫死本機真實照片路徑 | commander 於 D3b 裁定（`d-20260929193128013174-12` item 5）不併入 D3b：該檔有 skipif 與存在檢查、唯讀，且不在 D0–D4 範圍內 | 另案處理 |
| 12 | **launcher 行為變更會影響 D4 驗收可重現性** | D3b 已把 `scripts/g3-local-test-app.command` 由 record 模式改為 `--mode demo`（decision `d-20260929193651396921-13` item 4），而本表第 5 項已記錄 launcher 啟動時會自動 `git pull --ff-only` 快轉 main。D4 若要求固定版本驗收，launcher 本身已是會變動的因子 | **D4 前置 PR 已解**：launcher 不再 pull，改為顯示 commit 與落後狀態（decision `-14`），SOP 已同步。仍需真機確認顯示正確 |

| 13 | **D2b 引入的雙重釋放路徑在真機 `OpenCVCapture` 上未實測** | `cancel_clicked` 仍呼叫 `release_source()`，接著 `enter_ready` 現在也呼叫 `close()` —— 這條路徑在 D2b 之前不存在（當時 `enter_ready` 是 detach-only）。合成環境實測安全（`close()` 抽象契約明寫 `Idempotent`，`OpenCVCapture.close()` 以 `handle, self._handle = self._handle, None` 守衛使二次呼叫跳過；FakeCapture 實測 Cancel 後 `is_closed=True`、thread 數 `1 → 1` 不增長、相機可重新 Start），**但真機 AVFoundation 的 handle 釋放冪等性未經實測** | D4 實測 |

| 14 | **D4-F1 新增的 worker `except` 是防禦性 guard，無已知實例** | `controller.py` 的推論 worker 原本只有 `try/finally`、沒有 `except`；issue #123 暴露後補上的 `except Exception` 會把任何未預期例外轉成 `SessionStatus.error` 終局（複用既有狀態值，零新增文案）。**但它不是 #123 的修復**（根因在 `cmd_live` 的 demo 接線），而且**寫測試時量到：三個明顯失敗點裡有兩個早已被上游防禦** —— scorer 例外已轉成 `scorer_failure: <Type>` 終局（`controller.py:361`）、research sink 例外已記錄於 `_research_sink_errors`（D2 的設計）。修好 #123 之後，**已知會逃出 worker 的路徑幾乎為零**；仍存在的候選是 `frame_transform`（D2 刻意讓它 propagate 以 fail-closed），但沒有任何已知會觸發它的實例。**所以這個 `except` 防的是「未知的未知」，不是一個正在發生的問題 —— 後人不可據此認為它擋過某個真實故障。** 註：Reviewer 同意該做（成本實質為零），但同樣要求把它記錄在此 | 已落地並記錄（D4-F1） |
| 15 | **兩個 D4-F1 守護檔在任何 CI job 都沒執行過（B1）** | 首版 D4-F1 PR（`61558e0`）新增兩個守護檔，但 `ci.yml` 未列出：wiring 檔需要 `research-ui` extra **與本機 YuNet artifact**、worker 檔需要 `research-ui`（走 fixture 的 `importorskip`），而 `verify` job 只裝 `.[dev]`。實測兩個平台全部 8 個 skip、passed 數完全不變（base `1326d10` macos `879 passed, 107 skipped` → 本 PR `879 passed, 115 skipped`；ubuntu `878/108` → `878/116`，skipped 恰 +8）。**「守護存在、有 mutation 自證、報告寫著 failed」但從未在 CI 執行** —— 這與 D1–D4 各代假保護同族：證據看起來充分，但它覆蓋的環境小於它宣稱的 | 已落地（`qt-smoke`） |
| 16 | **`qt-smoke` 對外網路的相依是 D4-F1 新引入的** | B1 的修正必須讓 `qt-smoke` 取得 detector artifact（wiring 守護真的跑），該 workflow 原本零對外網路相依。已實測：`docs/research/2026-09-10-model-candidate-gate.md:23` 記錄的 opencv_zoo URL 重現凍結 SHA `8f2383e4…`，下載檔與本機 artifact 逐 byte 相同；job 內以 `sha256` 硬驗，不符即 exit 1（已用「HTTP 200 但內容被替換」的模擬 curl 驗證該守衛是唯一能攔住它的機制，exit 1）。**這是新增的失效面**：日後 opencv_zoo 不可用時該 job 會紅，與測試本身無關 | 已落地，待 D4 實測一併觀察 |
| 17 | **D4 真機第一輪：演算法首次在真機上跑出結果（2026-09-30，32 列）** ⚠️ **該批資料已於 2026-10-03 清除重跑 —— 本項的統計數字是 2026-09-30 的歷史觀測，不可再引用為現況。** | operator 在 main `dd5badd`（D4-F1 #123 修好**之後**）自測真機 32 輪，結果在 `store/demo-results.csv`。**這是 D0–D4 里程碑第一次有真機演算法結果**，但**單人、單一場景，不構成準確率**（第 4 項屬 D5）。實測彙總：`mode` 全為 `demo-no-recording`；`result` 為 `invalid_input` 16、`matched` 11、`timeout` 5；`label_kind` 為 `unlabeled` 12、`enrolled` 8、`uncertain` 7、`unenrolled` 5；**有 top1 的 16 列全部是 `enroll-23`，沒有任何一輪認成別人**。**matched 11 輪**：`top1_score` 0.6219–0.7279、`top2_score` 全檔最高 0.2801、`margin` 0.3692–0.5080、reason 全部 `supported_3_frames`。⚠️ **`margin` 下界 0.3692 只比 `match_threshold` 0.363 高 0.0062** —— 餘裕極窄，這對第 3 項的判讀有意義。**D3b 邊界確認：研究用 `results.csv` mtime 仍是 2026-09-29 15:20:18，未被寫入**，符合非錄製不落盤 | 已記錄（D4 收尾） |
| 18 | **⚠️ `unenrolled` 標註的語意陷阱：`label_kind` 與該列的分數可能相反（D5 必讀）** ⚠️ **同第 17 項：該批 log 已於 2026-10-03 清除重跑，本項的列數與分數是歷史觀測，不可再引用為現況。** | 該輪 5 列 `unenrolled` 中，**4 列的 `top1` 都是 `enroll-23`、分數 0.5621–0.6995，`margin` 0.3101–0.4279**，遠高於門檻 0.363／0.10（另 1 列是 `no_frames_captured`，沒拍到影格）。依 `qt_window.py:1467-1472` 的 docstring「**Correct endorses what is shown**: a matched round records the shown identity (operator-confirmed, kind enrolled); a **not-found round records unenrolled** (operator confirms absent)」—— 語意的白紙黑字宣告；實作在 `:1498-1505`（`if correct: identity = self.desktop.display_identity()` → `if identity is None: … kind="unenrolled"`） —— **而實際上這個人就是已註冊的本人。標註語意與觀察事實相反，不是記錄瑕疵。** 成因：那一輪系統回「找不到」，operator 照 UI 語意按「正確」，但 UI 沒有、也不該知道那個人其實在註冊組裡。**D5 若要用此檔建 truth mapping，這 5 列不可直接採用** —— 而它們正是最該被當成「已註冊者」正例的那幾輪 | 已記錄，待 D5 處理 |
| 18b | **第 18 項那 4 列失敗的機制是三幀規則的連續性條件，不是曝光** ⚠️ **同第 17 項：那 4 列出自已於 2026-10-03 清除重跑的批次，本項的實測值是歷史觀測。** | 實測該 4 列的 reason 為 `insufficient_evidence\|support_0_of_3`（3 列）與 `support_1_of_3`（1 列）。查 `session.py:620-651`：這個終局條件是 `_frames_usable > 0` 且 `len(_support_sequences) < required_support`，即**同一個人連續達標的幀數不足 3**。而每幀達標需同時過 `match_threshold`、`margin_threshold`，**以及 `min_support_interval_ms`（200ms）間隔**（`session.py:479-500`：前一幀不足 200ms 就 `interval_skip`，不累加；換人則整窗歸零重新計算）。**所以更可能是連續性／取樣間隔條件，而非畫面品質** —— 該輪 `required_support=3`、`sample_interval_ms=200`、`max_frames=26`。⚠️ **還有一條同樣能產生這個結果的歧義路徑**：`session.py:436-461` 顯示單幀若 `top_score < match_threshold` 或 `margin < margin_threshold` 會觸發 `score_reset` 並清空 support window —— 該輪的 `best_baseline` 分數雖高，但那是**單幀最佳值**，不保證每幀都達標，所以「幀夠多卻只累積到 1」也可能是 **support 幀之間夾雜低分幀**，而不只是取樣間隔。⚠️ **此推論僅由 reason code 與程式碼路徑得出，未看畫面、未看 `quality_rank` 分布**，`quality_rank` 排序本身也影響誰被當成最佳幀（`cli.py:293-296`）。**不得當成已確定的根因**；**D4 重跑時若能取到逐幀事件，取 `score_reset` 與 `interval_skip` 的計數即可一舉分辨這兩條路徑** | 待 D4 重跑時一併量 |
| 19 | **D2b 契約的真機落差：結果階段相機燈號熄滅、「再次辨識」不會立刻續跑（operator 指示暫不修）** | D2b（decision `d-20260929173733098323-10`、PR #121）交付的是「結果階段保留鏡頭」＋「再次辨識／停止相機」。**真機第一輪觀察到此契約未達成**：辨識結束後**相機燈號會熄滅**；按「再次辨識」**不會立刻重新開始 preview 與辨識**，實際流程只能回到 Start。⚠️ 這是**真機與 D2 合成驗收的落差**（合成環境 `FakeCapture` 下已實測連續預覽路徑安全，見第 13 項）。**operator 明確指示暫不修**，故此處**只記錄不動工**；`src/` 零改動。**下一輪 D4 重跑時請先觀察這兩點是否再現** | 已記錄，暫不修（operator 指示） |

**D2b 對第 12 項的影響（2026-09-30，decision `d-20260929173733098323-10`）：** 連續預覽使 operator 得以連續重試，**每輪不再必然以標註收尾**。因此 SOP 已同步更新，明確記錄 `label_kind=unlabeled` 這個 demo 專用值代表「跑過但沒標註」，且它不影響研究用的 `results.csv` 列舉與 `report.py`／`analysis.py` 的統計分類（決策 `-11`）。D4 若以 demo 模式做固定版本驗收，**必須先決定 `unlabeled` 列計入分母的規則**，否則逐輪數與已完成測試數會不一致 —— 這是 D2b 新增的驗收前提，第 5、12 項的 pull 凍結仍須一併處理。

**第 3 項的樣本口徑更正（2026-09-30，D4 真機第一輪實測後）：** 「曝光或品質拒絕」的樣本**不是** `insufficient_evidence` 那幾輪 —— 那一類的原因是三幀規則的**連續性**條件（`min_support_interval_ms=200ms`），與曝光無關，見第 18 項的機制說明。真正對應「拍了影格但幀被拒」的是 `all_frames_rejected_*`，該輪共 **5 列**。

**本清單不授權任何相機操作。** 真機測試需 operator 在場並另行授權（`--device local` 未授權）。

## 12. 邊界

- 本檔不宣稱 D1–D4 任何一項已完成，也不宣稱真機辨識成功。
- 照片、embeddings、DB、逐輪結果、加密診斷與設定檔**一律不進 Git**（`.gitignore` 已擋圖片／資料庫）。
- 既有 `results.csv` 與加密檔**唯讀**；本 D0 全程未覆寫或刪改（fake 重現使用暫存目錄）。
- 歷史 G3 採集資料**可用於故障診斷，不自動視為有效準確率樣本**。
- 對外回報不貼完整 uid。
