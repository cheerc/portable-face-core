# D5 現版本靜態 baseline 執行計畫

> 拋棄式執行文件：執行完不維護、不回填。功能與研究結論的權威是報告本身與 `docs/PROJECT-STATE.md`，不是本檔。

**Source of truth:**
- 目標與統計欄位：`PLAN-mac-face-demo-handoff.md` §4 D5（repo 外的 operator 交接文件）。原文：「用現版本對 13 張辨識組與 30 張 non-target 建 baseline；先確認真實身份 mapping。分開統計 top1 排序、正確接受、錯誤接受、未知拒絕、品質拒絕、無法處理及延遲，不把拒絕樣本無聲排除。資料小且曾被檢視，不當獨立泛化證據。」
- 開工授權與辨識組 truth：decision `d-20260930051029074216-21`（operator：「辨識組 13 張 全部都是我，都是註冊組的 enroll-23 沒錯。可以派D5開工」）。
- non-target truth：decision `d-20260930051401221136-22`（operator：30 張「都不是註冊組的人」）。這份 decision supersede `-21`，是目前的 active leaf，task 的 governing decision 要用它；`-21` 的開工授權與辨識組 truth 繼續有效。

**Base:** `cheerc/portable-face-core` main `dd5badd8db258b81b534eb9f9f9bf552bebcc80b`。lead 開工時要重查 HEAD，不能把這個 SHA 當成一定是最新。

**Goal:** 用 **App 現場實際在用的同一條 pipeline**（同一個 gallery 建構、同一個 detector gate、同一套品質篩選、同一組 g3-v1 門檻），對 13 張辨識組和 30 張 non-target 逐張評分，得到一份之後換模型（D6）可以直接拿來對照的基準線。

**Architecture:** 新增一個評估 harness `src/facecore/eval/static_baseline.py`。它直接呼叫 live 路徑現有的 `_build_true_context`（`src/facecore/research/cli.py:623`）和 `score_frame`（`src/facecore/live/frame_pipeline.py:366`），不另外建第二條 scorer/policy pipeline。每張照片輸出兩個分支：
- **L 分支**：App 看到的結果，會經過品質篩選。
- **R 分支**：不經品質篩選，只看模型排序。

兩個分支分開，才分得出「模型認不出」和「品質篩選擋掉」。逐張明細寫到 repo 外；repo 只放彙總報告。

**Tech stack:** Python 3.14、uv、onnxruntime、YuNet 2023mar、SFace 2021dec fp32、pytest、ruff、mypy。

## Global Constraints

- 不換模型，不改 detector gate，不改 profile `g3-v1`（match 0.363／review 0.3／margin 0.10）、三幀規則、gallery 或 learning。門檻只能用 sweep 呈現，**不得選定新門檻**：這批資料沒有 holdout，事後從 grid 挑門檻等於用測試集調參。
- 不開相機。D5 是靜態照片評估。
- 真實照片、embeddings、DB、逐張原始輸出**不進 Git**（`CLAUDE.md:18`、`docs/PROJECT-STATE.md:116`）。repo 內的報告只放計數與分數區間，並且必須通過 `src/facecore/eval/report.py` 的 `write_report` 遮罩檢查（會拒絕 float 向量、絕對路徑、`identity: enroll-` 這類逐張身分標籤）。
- 既有的 `results.csv`、`demo-results.csv` 和加密檔都只讀。
- 不修改 M1–M5 用過的既有評估模組（`eval/nontarget_fa.py`、`fa_matrix.py`、`sweep.py`、`bakeoff.py`、`real_replay.py`），也不修改 `src/facecore/live/`、`src/facecore/research/cli.py` 的行為。這樣歷史報告可以重現，D0–D4 已交付的路徑也不會被動到。
- 靜態單張的結果**不得推論動態 session 的表現**，反過來也一樣（`docs/PROJECT-STATE.md:114`）。三幀規則不適用於單張照片，報告要明寫這一點。
- 資料量小（13＋30），而且曾經被檢視過，**不得當作獨立的泛化證據**。
- commander 只交付 plan，不做 repo mutation。lead 負責落地與收斂。

## Evidence and Open Hypotheses

**Established（commander 在 `dd5badd` 親驗，或由 source 盤點取得並附行號）：**

- **在 repo 內沒有任何工具能讓照片走 App 的同一條路徑。**
  - 既有 `cmd_bakeoff` 的 probe 路徑是 decode → detect → align → embed，**沒有品質篩選**（`src/facecore/cli.py:267-280`）。band 的判定用 `REVIEW_THRESHOLD=0.5`（`cli.py:144,285-292`），和 live 的 review 0.3 不同。
  - 目標身分寫死成 `"person-23"`（`cli.py:302,478`；`eval/fa_matrix.py:129`），但 live gallery 是用檔名 stem 當身分，也就是 `enroll-23`。
  - `cmd_evaluate` 不會評分（`cli.py:92-93`）。
- **live 路徑可以重用。**
  - `_build_true_context(models, corpus, profile, *, gallery_dir=...)`（`research/cli.py:623-700`）會用 `build_gallery_from_folder`（generation `gen-1`）加上 `profile_to_policy(profile)`（`:733-744`）組出 `ScoringContext`。
  - `_load_profile(profile_path)` 在 `research/cli.py:223`。
  - g3 設定檔的載入是 `load_g3_config`（`research/g3_config.py:32`）。
  - `score_frame(frame: FramePacket, context, *, diagnostic_sink=None)`（`frame_pipeline.py:366`）一次做完 detect、品質篩選、embed、比對。`stage_durations_ms` 會記錄 detect、quality、embed 各段的毫秒數，但要設 `diagnostic_sink` 才會輸出。
  - `FramePacket(sequence>=1, captured_ns>=0, rgb=np.ndarray, orientation=0, mirrored=False)`（`live/contracts.py:22-44`）。
- **品質篩選沒過就沒有分數。** `score_frame` 在 quality 未通過時回傳 `identity_scores={}`（`frame_pipeline.py` quality 段，`scoring_missing_reason="quality_rejected"`）。所以 L 分支拿不到被篩掉照片的分數，需要 R 分支補上。
- **單幀 band 的判定**：live 是 `score ≥ match 且 margin ≥ margin → matched`；`score ≥ review → review`；其他 → unknown；沒有 runner-up → unknown（`live/session.py:772-784`）。
- **品質篩選的門檻**（`contracts/policy.py:35-43`，`frozen_v1`）：detector confidence ≥ 0.90、臉短邊 ≥ 112 px、清晰度 ≥ 60、平均亮度 40–215、過曝或過暗像素比例 ≤ 0.05、yaw ≤ 30°、pitch ≤ 20°。
- **既有 `sweep.sweep_thresholds` 的 FA 定義只看 score ≥ match，不看 margin**（`eval/sweep.py:50-52`），和 live 的 matched 定義不一致。D5 的 FA 必須採用 live 定義。
- **D0 凍結值**（`docs/mac-demo-baseline-d0.md`）：
  - gallery digest `e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f`，23 個身分，generation `gen-1`（`:74-77`）。
  - YuNet sha256 `8f2383e4…52fa4`、SFace sha256 `0ba9fbfa…4e79`（`:37-38`）。
- **EXIF**：live gallery 解碼時沒有做 EXIF 旋轉（`frame_pipeline.py:210-213`）。commander 只讀了中繼資料（沒有解碼像素）：註冊組 23 張的 orientation 是 21 張無標記、2 張 =1；辨識組 13 張、non-target 30 張都沒有標記。**現有資料都不需要旋轉，這個差異對 D5 沒有影響**，只列為潛在風險，不在本計畫處理。
- **歷史數字不可比。**
  - M1（FA 4/30）、M5 是在 `ALIGN_CONTRACT_VERSION=1`、detector gate 0.8 下量的（`docs/PROJECT-STATE.md:104-110`）。
  - v3 重跑（M1 1/30；M5 margin 0.15 match 0.30–0.45 → 13/13 | 1/30）只存在 board task `t-20260915125726513251-42526-1` 的 result 裡，沒有留下逐張輸出，gallery digest（`0e78f20f…`）也和 live 的不同。
  - 報告可以引用這些數字作為歷史脈絡，但**不能把它們和 D5 並列成同一協議下的比較**。
- **D4 動態觀察**（`demo-results.csv`）：operator 按「正確」的 8 輪全部 matched `enroll-23`，top1 0.62–0.73、top2 0.17–0.27、margin 0.40–0.51。這是動態的 session 結果，報告只能另列一段「動態觀察」，不得併入靜態分母。另外有 5 列 `label_kind=unenrolled`，實際上是 enroll-23（operator 在 not-found 結果上按了「正確」），**不得作為任何 truth 或分母**。

**Unproven（owner＝lead／impl，在 Task 1 開頭解決）：**

- H1：用 `_build_true_context(..., gallery_dir=註冊組)` 在本機建出來的 gallery，digest 會等於 D0 的 `e3d77c4c…`。**如果不等，就停下回報**（見 Stop conditions），因為那代表 harness 和 App 看到的不是同一個 gallery。
- H2：靜態照片用 `Image.open(p).convert("RGB")` 轉成 `np.ndarray`，再包成 `FramePacket` 餵給 `score_frame`，不需要改 live 程式碼就能跑。**如果做不到，就停下回報**。
- H3：R 分支（跳過品質篩選）可以直接呼叫 `context.detector.detect`、`align_crop`、`context.embedder.embed` 和 cosine 比對來組出來，不需要改 live 模組。R 分支的比對函式必須和 `score_frame` 用的是同一個；impl 要在 PR 裡寫出實際用的函式名稱和行號。

## Documentation impact check — `arch=N adr=N area=Y`

- `area`：`docs/PROJECT-STATE.md` 目前沒有 D0–D5 的現況（D0 更新過，D1–D4 之後沒有再同步）。D5 會新增一份 baseline 報告和狀態。這屬於 area 文件變更，不改架構，也不需要新的 ADR。
- **blocking dependency**：lead 在 Task 1 開工前執行 `project-docs-maintain` check，確認 PROJECT-STATE 要改哪些段落，並把結論寫進 task。實際的文件更新和報告一起在 PR-B 落地（內容取決於實測結果，不能提前寫）。PROJECT-STATE 不得把還沒做的東西寫成已驗證。
- 如果 check 發現還有 arch 或 adr 的影響，先停下，回報 commander。

## File／Interface／Dependency Map

| 位置 | 動作 | 責任 |
|---|---|---|
| `docs/plans/2026-09-30-d5-static-baseline-plan.md` | Create（PR-A） | 本計畫落地，照 byte-for-byte 從 commander workspace 複製 |
| `src/facecore/eval/static_baseline.py` | Create（PR-A） | harness：載入 context、逐張 L／R 評分、彙總、sweep、輸出 |
| `tests/eval/test_static_baseline.py` | Create（PR-A） | 只用 synthetic detector／embedder／小圖；不需要真實權重與照片 |
| `.github/workflows/ci.yml` | 視情況 Modify（PR-A） | 如果新測試在 `verify` job 會被 skip，就要加進有跑的 job。判準是「實際 N passed」，不是 CI 綠燈（D4-F1 B1 的教訓） |
| `docs/research/2026-09-30-d5-static-baseline.md` | Create（PR-B） | 彙總報告（只放計數與分數區間），用 `write_report` 寫出 |
| `docs/PROJECT-STATE.md` | Modify（PR-B） | 把現況同步到 D0–D5；D5 baseline 的數字與限制 |
| repo 外：`~/Downloads/face_sample/_facecore/reports/d5-static-baseline-<sha7>.csv` | 真實跑時產生 | 逐張明細，**不進 Git** |

**Interfaces（PR-A 產出，PR-B 使用）：**

```python
# src/facecore/eval/static_baseline.py
@dataclass(frozen=True)
class PhotoResult:
    set_name: str              # "probe" | "nontarget"
    photo: str                 # 檔名（不含目錄）
    truth_identity: str | None # probe="enroll-23"，nontarget=None
    face_count: int
    unprocessable_reason: str | None   # 無臉、多臉、解碼失敗等，沿用 score_frame 的 reason
    quality_pass: bool
    quality_reasons: tuple[str, ...]
    # L 分支：App 路徑，品質沒過時全部為 None
    l_top1: str | None; l_top1_score: float | None
    l_top2: str | None; l_top2_score: float | None
    l_margin: float | None
    l_band: str                # "matched" | "review" | "unknown" | "rejected"
    # R 分支：跳過品質篩選的模型排序；沒有單一人臉時全部為 None
    r_top1: str | None; r_top1_score: float | None
    r_top2: str | None; r_top2_score: float | None
    r_margin: float | None
    r_band: str | None         # 用同一套 live band 規則算
    stage_ms: dict[str, float] # detect／quality／embed，來自 diagnostic_sink

def score_photo(path: Path, context: ScoringContext, profile: ResearchProfile,
                *, set_name: str, truth_identity: str | None, sequence: int) -> PhotoResult: ...
def summarize(results: list[PhotoResult]) -> BaselineSummary: ...   # 各類計數＋分數區間
def sweep_live_semantics(results: list[PhotoResult], match_grid: list[float],
                         margin_grid: list[float], review: float) -> list[SweepCell]: ...
def main(argv: list[str] | None = None) -> int: ...                 # python -m facecore.eval.static_baseline
```

**統計定義（沿用 live 語意，寫死在 harness 內，並用測試釘住）：**

- 辨識組（truth = enroll-23）：
  - 正確接受：band=matched 且 top1=enroll-23
  - 認錯人：band=matched 且 top1≠enroll-23
  - review：band=review
  - 未知拒絕：band=unknown
  - 品質拒絕：quality_pass=False
  - 無法處理：face_count≠1 或解碼失敗
  - top1 排序正確：top1=enroll-23，不管 band，L、R 各算一次
- non-target（truth = 未註冊）：
  - 錯誤接受（FA）：band=matched，也就是 score ≥ match **且** margin ≥ margin。這是 live 的定義，**不是** `sweep.py` 那種只看 score 的定義。
  - 正確拒絕：band=unknown
  - review：band=review
  - 品質拒絕、無法處理：定義同上
- 每一類都要寫出分母（k/N）。分母是照片張數，被拒絕的照片**不得從分母中移除**。
- sweep grid 沿用 `REAL_MATCH_GRID`（0.30–0.60）與 `REAL_MARGIN_GRID`（0.05–0.20）（`eval/nontarget_fa.py:52-55`），用 import 取值，不要複製常數。兩個分支、兩個臂都套用同一套 live 規則。

**Dependency order：** PR-A（Task 1：harness＋synthetic 測試＋plan 落地）merge 之後 → Task 2 在本機跑真實照片 → PR-B（Task 3：報告＋PROJECT-STATE）。

## Task 1：harness 與統計定義（PR-A）

**Owner:** impl（lead 派工）
**Files:** Create `src/facecore/eval/static_baseline.py`、`tests/eval/test_static_baseline.py`；視情況 Modify `.github/workflows/ci.yml`。

**Consumes:** `_build_true_context`、`_load_profile`（`research/cli.py`）、`load_g3_config`（`research/g3_config.py:32`）、`score_frame`、`FramePacket`、live band 規則（`live/session.py:772-784`）、`REAL_MATCH_GRID`／`REAL_MARGIN_GRID`、`write_report`。

**Produces:** 上面列出的 interfaces。

**開工前（PRE_WORK）必做：** 確認 H2、H3 成立。如果呼叫這些函式需要改 live 或 research 模組的行為，就停下回報。如果只是想把私有函式（`_build_true_context`、`_load_profile`）公開化，也要先回報 lead，由 lead 決定；預設做法是直接 import 私有函式，並在 harness 裡加註解說明原因。

**Test-first evidence（全部用 synthetic 的 fake detector／embedder，不需要權重）：**

1. band 定義：
   - 構造 top1=0.50、top2=0.45（margin 0.05 < 0.10），預期 `l_band="review"` 而不是 matched。
   - 構造 top1=0.40、top2=0.20，預期 matched。
   - 構造 top1=0.32，預期 review（因為 ≥ 0.3）。
   - 構造 top1=0.29，預期 unknown。
   - RED：`uv run --extra dev python -m pytest tests/eval/test_static_baseline.py -q`，在 harness 還不存在時必須是 ImportError 或 FAIL。
2. 品質拒絕的照片：L 分支全部為 None、`l_band="rejected"`，R 分支仍然有分數。
3. FA 定義：non-target 的 top1=0.40、margin=0.05，**不算** FA（sweep.py 的定義會算，這裡要證明兩者有差別）。
4. 拒絕樣本留在分母：3 張照片裡 1 張無臉，summary 的分母是 3，不是 2。
5. 認錯人：probe 的 matched 但 top1≠truth，要計入「認錯人」，不能計入「正確接受」。
6. 遮罩：用 `write_report` 寫出 summary 時，不得觸發 `RedactionError`；把絕對路徑塞進 body 時，必須觸發。

**Mutation 自證（必做，D2–D4 已經連續五代出現同型的假保護）：** 至少做下面三個，每個都要附上「改壞 → RED、改回 → GREEN」的實際輸出：
(a) 把 FA 判定改成只看 score；
(b) 把被拒絕的樣本從分母中移除；
(c) 把 R 分支改成也經過品質篩選。
斷言的對象必須是 harness 實際會走到的路徑。

**GREEN／Project verification：**

```bash
uv run --extra dev python -m pytest tests/eval/test_static_baseline.py -q
uv run --extra dev python -m pytest tests/ -q
uv run --extra dev --extra research-ui python -m pytest tests/ -q
uv run --extra dev ruff check src tests
uv run --extra dev mypy src
```

判準要以 CI log 裡的**實際 passed 數**來看：新測試在 CI 的哪個 job 跑了、跑了幾個。不要只看綠燈。

**Acceptance（PR-A）：**
- 上述 6 類測試加 3 個 mutation 全部成立。
- 兩種依賴組合都通過，而且在 `dev` 組合下新測試**不是 skipped**。
- `git diff origin/main -- src/facecore/live src/facecore/research src/facecore/eval/{nontarget_fa,fa_matrix,sweep,bakeoff,real_replay}.py` 為空。
- plan 檔已落在 `docs/plans/`。

## Task 2：真實照片本機執行（PR-A merge 後；不需要 PR）

**Owner:** impl，在 operator 這台 Mac 本機上跑，唯讀讀取 operator 提供的照片資料夾。不開相機。

```bash
uv run --extra dev python -m facecore.eval.static_baseline \
  --config ~/Downloads/face_sample/_facecore/g3-local.json \
  --probes ~/Downloads/face_sample/辨識組 --probe-truth enroll-23 \
  --nontarget ~/Downloads/face_sample/non-target \
  --detail-out ~/Downloads/face_sample/_facecore/reports/d5-static-baseline-<sha7>.csv \
  --report-out docs/research/2026-09-30-d5-static-baseline.md
```

（旗標名稱由 impl 在 Task 1 定案，語意要符合上面這條命令，並在 PR-A 裡寫明。）

**執行前核對（任一項不符就停下回報）：**
- gallery 23/23 載入。
- gallery digest = D0 的 `e3d77c4c…`（H1）。
- 兩個模型的 sha256 = D0 凍結值。
- profile = g3-v1。
- 辨識組確實是 13 張，non-target 確實是 30 張。

**Acceptance：**
- 逐張明細 CSV 產生在 repo 外，共 43 列。
- 彙總報告通過 `write_report` 遮罩。
- `results.csv`、`demo-results.csv` 的內容 hash 在執行前後相同。

## Task 3：彙總報告與 PROJECT-STATE（PR-B）

**Files:** Create `docs/research/2026-09-30-d5-static-baseline.md`（由 Task 2 產生）；Modify `docs/PROJECT-STATE.md`。

**報告必須包含：**
1. 版本資訊：`ALIGN_CONTRACT_VERSION`、detector gate、profile、gallery digest、model generation、兩個模型的 sha256、HEAD。
2. 辨識組 13 張、L 分支：正確接受／認錯人／review／未知拒絕／品質拒絕／無法處理（k/13），以及 top1 排序正確數；R 分支同樣列出。
3. non-target 30 張、L 分支：FA／review／正確拒絕／品質拒絕／無法處理（k/30）；R 分支同樣列出。**另外單獨列出現行門檻 0.363／0.10 下的 FA**，因為這是 App 現場真正在用的門檻。
4. 分數區間：辨識組 top1、non-target top1、兩組 margin 的最小值、中位數、最大值。**不寫逐張分數**。
5. 品質拒絕原因的分布（依 reason code 計數）。這是 D0 §11 第 3 項（曝光）的靜態證據。
6. 延遲：detect／quality／embed 各段毫秒數的中位數與範圍。註明「靜態照片的解析度和相機畫面不同，這組數字不代表現場延遲」。
7. sweep grid（live 語意，只作呈現，不選門檻）。
8. 限制段：資料小、曾被檢視過、不得當作泛化證據；靜態結果不推論動態；歷史 M1／M5／v3 數字的協議不同（gate、contract、gallery digest 都不同），只能當脈絡、不能直接比；D4 動態觀察另列。

**PROJECT-STATE：** 把現況同步到 D0–D5，加入 D5 baseline 的指標與限制，並指向報告。不得把還沒做的東西寫成已驗證。

**Acceptance（PR-B）：**
- 報告內的數字和 Task 2 輸出一致（reviewer 用 Task 2 的 CSV 重新計數核對）。
- `write_report` 遮罩通過。
- `git diff` 只動到報告與 PROJECT-STATE。
- CI 通過。

## PR 邊界

| PR | 內容 | 自己的 acceptance |
|---|---|---|
| PR-A `feat(eval): D5 static baseline harness` | Task 1＋plan 落地 | 只用 synthetic 測試＋mutation；live／research／歷史評估模組零 diff |
| PR-B `docs(research): D5 static baseline report` | Task 3（依 Task 2 的輸出） | 報告數字可由 repo 外的 CSV 重算；PROJECT-STATE 不越權 |

PR-A 是給後面使用的 foundation，本身不含任何真實資料結果，所以可以獨立 merge。PR-B 依賴 PR-A 和 Task 2。

## Stop conditions（任一項成立就停下，回報 lead 再轉 commander）

- gallery digest ≠ `e3d77c4c…`，或 23 張沒有全部載入。
- 靜態照片要經過 `score_frame`，必須修改 live 或 research 模組的行為（H2），或者 R 分支必須修改 live 模組（H3）。
- 要得到有意義的結果，必須改門檻、換模型、改 detector gate，或跳過三幀規則以外的 live 規則。
- `write_report` 遮罩擋下彙總報告，而且唯一的解法是放寬遮罩。
- 任何步驟需要開相機，或需要把照片、embeddings、逐張明細放進 repo。
- `results.csv` 或 `demo-results.csv` 在 Task 2 前後的 hash 不同。

## Highest risks

1. **又一個假保護**：band／FA 的測試只測 harness 自己寫的 helper，沒有測到 `score_frame` 真正的輸出路徑。對策：Task 1 的 mutation (a)–(c) 必做，reviewer 要獨立重跑。
2. **把 R 分支當成 App 的結果**：R 分支跳過了品質篩選，會比 App 樂觀。對策：報告以 L 分支為主，R 分支標明「僅供模型排序參考，App 不會這樣判定」。
3. **歷史數字的誤比較**：M1 4/30、v3 1/30 和 D5 的 FA 放在一起看，很容易被讀成「變好了」或「變差了」。對策：報告的限制段明寫三者的協議差異，不做同表並列。
4. **樣本太小**：13 張全部來自同一個人。任何比率都不能外推到其他人或 500 人規模。

## 報告結論後的下一步（不在本計畫範圍）

D6 用同一個 harness 換上其他模型後端，在同一份 truth mapping 下比較。要換哪些模型，由 operator 另外決定。
