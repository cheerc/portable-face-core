# D6 DeepFace 模型比較計畫

**狀態：** 已完成資料蒐集的 plan 草案，供 operator 審閱。沒有派工、安裝新相依、下載新權重或執行 DeepFace。這不是產品替換、門檻選定或部署授權。
**Source of truth：** operator 選 `serengil/deepface`、L／R「都看」、A＋C 並行，以及「先寫 D6 的 plan」；truth mapping 為 decisions `d-20260930051029074216-21`／`d-20260930051401221136-22`。既有研究邊界見 `docs/research/2026-09-09-open-source-face-stack.md:67-74`、`CLAUDE.md:18-24`。
**Base full HEAD：** `592493968eb5f47bbd37b57d4e1f275330e7a56c`（D5 PR #127）；本輪核對 local `origin/main` 與 GitHub main 相同。正式接工時 lead 再核對相關 diff。
**DeepFace source pin：** `e1b38a8fced707b8acb137bcc0052302a3cae329`。下文上游來源皆固定此 SHA，不用持續漂移的 `master` 作執行契約。
**Goal：** 用相同 23 張註冊照、13 張辨識照、30 張未註冊照與明確前處理，比較現有 SFace 與 DeepFace 候選的辨識排序、正確接受、錯誤接受及成本；同時量出品質拒絕的數值原因，不改現有 App。
**Architecture：** 復用 D5 harness 的 `ScoringContext`／`embedder_factory` 接縫，新增 research-only adapter。固定 YuNet、align3、品質 policy 與照片集合；模型必要的 resize／色彩／normalization 各自明列。DeepFace 放在獨立實驗環境，與產品 Python 3.14／ORT 相依隔離。
**Tech stack：** 產品側 Python 3.14、uv、ONNX Runtime、numpy、Pillow、pytest／ruff／mypy；DeepFace 側 Python／TensorFlow 或 PyTorch 的 exact lock 由 compatibility spike 實測凍結，不提前宣稱相容。

## 1. 比較設計與範圍

採用 **固定 detector／landmark alignment，只換辨識 backend**，不把偵測、對齊、品質 gate、模型、門檻全部換掉後統稱為「演算法差異」。

| 方案 | 本計畫處理 |
|---|---|
| 固定 YuNet＋align3，換 embedder | 採用；主要比較模型及其必要前處理 |
| DeepFace 預設完整 pipeline | 不做；它同時改 detector／alignment／threshold，無法隔離差異 |
| 直接替換 App 模型 | 不做；比較不是 runtime migration 或 mobile 交付 |

共同來源 crop 是既有 `align_crop` 輸出的 112×112 RGB（`pipeline/align.py:25-26,90-147`）。不同模型需要不同尺寸時，允許 adapter 做必要 resize，但必須記錄，不能宣稱前處理完全相同。固定 112 crop 放大到 Facenet512 的 160 輸入，是實驗限制；本結果不代表 DeepFace 原生完整 pipeline 的最佳效果。

C 的品質診斷與 A 的候選／runtime 查證可並行；正式比較報告必須引用兩邊 evidence。C 不改任何行為，A 不回寫產品門檻。

## 2. 事實、推論與先前更正

- 註冊路徑不執行品質 gate、probe 路徑執行，是已查證的程式事實；**不對稱本身不等於 bug**。operator 認為同來源照片品質可用，應保留這項觀察；同來源不單獨證明每張照片的 luma 相同，也不直接證明 gate 錯誤。
- D5 的 `quality_exposure` 可能來自平均 luma 低於 40、高於 215，或 clipped fraction 大於 0.05。**不能將 23 張全部說成「太暗」**（D5 報告 §5、`pipeline/quality.py:41-46`）。C 要分開三種條件，允許同時觸發。
- L2 normalization／cosine 不會讓不同模型的 score 分布相同。新模型不能直接套用 SFace 的 match 0.363／review 0.3／margin 0.10。
- DeepFace 是框架，不是單一模型。不能把 InsightFace 某個權重的限制套到 DeepFace 同名 ArcFace，也不能把現有 SFace 一概寫成「僅非商用」。本 repo 的 SFace candidate gate 記錄 Apache-2.0 目錄聲明、commercial grant 與未解訓練 provenance（`docs/research/2026-09-10-model-candidate-gate.md` §1B）。
- 高相似度不證明「撞臉」、quality rejected 不證明照片有問題、R 正確不證明整體演算法無問題。報告只陳述本批資料的分布，不冒充成因診斷。

## 3. Global constraints

1. 不改 App、Qt、錄製／刪除、three-frame session、learning、model migration；「再次辨識」依 operator 指示暫不處理。
2. 不修改 root `pyproject.toml`／`uv.lock`、`src/facecore/live/`、`src/facecore/research/`、`profiles/g3-v1.json`、歷史 M1–M5 模組與 D5 報告。eval 區可新增研究入口；D5 CLI 預設行為要保留。
3. 不開相機、不連雲端人臉 API、不呼叫 `DeepFace.analyze` 或做年齡／性別等屬性分析。
4. 照片、crop、embeddings、identity DB、逐張可識別結果留 repo 外。預設不另存 crop 或 embeddings；本機跨 process 資料只走受控 pipe／記憶體，不進一般 stdout、log、GitHub 或 Artifact。
5. `results.csv`、`demo-results.csv`、D5 原始明細只讀；前後核 hash。mtime 只作附加證據。
6. 每個 backend 重建自己的 23 人 gallery，禁止新模型 probe 與 SFace 向量混比。原始身分／檔案 hash 相同，embedding digest 則按模型另外記錄；新模型不得要求等於 SFace digest。
7. 正式照片推論時不得 auto-download。先預備並驗 exact artifacts；empty cache 必須 fail-clear，不可靜默上網。
8. 這是小樣本靜態比較，不是 authentication、安全認證或代表性準確率。13 張全是同一人；失敗／拒絕不能從 13／30 分母消失。
9. commander 只交 plan／spec 內容，repo mutation、commit、review 與 merge 由 lead 承載。此輪不派工。

## 4. 已查證的接入契約

### 4.1 本 repo（base `5924939`）

| 項目 | Evidence／影響 |
|---|---|
| Embedder | `pipeline/embed.py:28-58`：`model_version: str`；`embed(crop: AlignedCrop) -> tuple[np.ndarray, str]`；SFace 輸出 L2-normalized |
| 注入 | `research/cli.py:623-700` 的 `_build_true_context(..., embedder_factory=...)`；D5 `eval/static_baseline.py:682` 目前不傳 factory |
| Gallery | `live/frame_pipeline.py:58-113`：immutable embeddings；digest hash sorted identities＋vector bytes，故需另記 model ID／weight hash／generation |
| 模型隔離 | `ScoringContext.__post_init__`（`frame_pipeline.py:126-137`）及 `score_frame` 的 version check（`:531-535`） |
| 比對 | `policy/identify.py:26-30` 的 `cosine_score`；不能只因兩個模型都512維就允許混比 |
| L／R | `static_baseline.py:300-361` 的 `score_photo`；L 呼叫 `score_frame`，R `_r_branch`（`:238-284`）跳過 quality |
| Band | `static_baseline.py:173-199`：score ≥ match 且 margin ≥ margin 才 matched；否則 review／unknown；無 runner-up 不 matched |
| 診斷 | `live/contracts.py:428-452` 的 `FrameDiagnostics`；`frame_pipeline.py:450-526` 發布 luma／clipped／sharpness／timing |
| 量測 | `pipeline/measure.py` 的 `exposure_of`、`sharpness_of`、`pose_of`；C 直接復用，不另寫近似算法 |
| 相依 | root 要求 Python `==3.14.*`，沒有 DeepFace／TensorFlow／PyTorch |

D5 control 必須重新核對：gallery23／23、digest `e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f`；YuNet SHA `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`；SFace SHA `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`；g3-v1、align3、detector gate0.90。

預期對照：L probe correct12／13、wrong0、quality1；L non-target FA2／30、review6、quality22；R probe correct13／13、wrong0；R non-target FA2／30、review28。若同版本、資料與參數卻不同，先查 divergence，不以舊CSV冒充本次control。

### 4.2 DeepFace（固定 `e1b38a8…`）

- `DeepFace.py:448-473`：ndarray 的 API 契約是 **BGR**；`represent` 有 `detector_backend`、`align`、`normalization`、`l2_normalize`。
- `modules/representation.py:118-203`：`skip` 不呼叫 detector／alignment，但仍 resize／normalize；skip 與共同分支各反轉一次通道，故 adapter 應先把來源 **RGB轉BGR**。`align=False` 不代表禁止 resize。
- `modules/preprocessing.py:10-113`：`base` 不另外做模型 normalization；`Facenet` 是 per-image mean/std；`Facenet2018` 是127.5縮放；`ArcFace` 是 `(x-127.5)/128`。**不會按 model_name 自動選。** resize後先有 `[0,1]` 尺度，max≤1的輸入另有行為，需用合成暗色塊測邊界。
- SFace：112／128維，OpenCV `feature`；Facenet512：160／512維；ArcFace：112／512維。H5與PyTorch `.pth` 是不同artifacts，不能只記模型名。
- `setup.py:60-66` 實際讀 `requirements.txt`，其中仍包含TensorFlow／Keras。`[pytorch]` extra 不會刪除 base TensorFlow 相依。不能沿用先前「只裝PyTorch就沒有TensorFlow」的說法。
- `commons/weight_utils.py:23-92` 只看cache是否存在，沒有checksum／offline開關；`folder_utils.py:7-34` 由 `DEEPFACE_HOME` 決定 `.deepface/weights`，import也會建目錄。adapter需在import前設定隔離home並驗cache，再阻擋推論時網路。

**比較正規化暫定：** SFace bridge=`base`；Facenet512=`Facenet`；ArcFace備選=`ArcFace`。這是待 E 查證的模型輸入契約，不是調參候選：E 必須找到exact weight對应訓練／reference推論依據，不能因名字相近就採用。不一致則停下回報，不掃三種 normalization 挑最好的一種。

## 5. 候選與授權／artifact gate

建議最小集合（operator 尚未逐一批准 individual weight）：

| Candidate | 用途 | 條件 |
|---|---|---|
| `ort-sface-control` | D5 control | 既有凍結artifact |
| `deepface-sface-bridge` | 同模型不同wrapper對照，抓色彩／尺度接錯 | 同一SFace權重bytes，預處理與數值差異記錄 |
| `deepface-facenet512` | 建議第一個不同recognition模型 | exact weight rights／provenance／runtime確認 |
| `deepface-arcface` | 第二候選備選，不預設必跑 | exact H5或pth資格核對；不套Buffalo_L授權 |

每個registry entry記 upstream commit、package version、engine、weight immutable locator、預期SHA來源／actualSHA、bytes、code license、weight license、允許用途／redistribution、training provenance已知與未解、Python/runtime/lock hash、color/shape/normalization/dim、generation。

已找到但**未下載核算**的metadata：
- SFace：OpenCV Zoo `47534e27c9851bb1128ccc0102f1145e27f23f98` Git LFS pointer，SHA與既有D0一致；目錄README明載all files Apache-2.0，訓練資料權利仍另核。
- Facenet512 H5：DeepFace release `v1.0/facenet512_weights.h5`；HF revision `897aaade186abd2b9aeba45038a27d16c2df5f4e` metadata SHA `3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f`。
- ArcFace H5：同release `arcface_weights.h5`；同HF revision metadata SHA `6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93`。
- HF mirror與GitHub release bytes是否相同未證實；兩個H5的exact license／轉換來源鏈未證實。wrapper MIT不補足此缺口。

下載前按 `CLAUDE.md:21` review並記錄。metadata缺口要明示與請operator裁定，不自行下載驗證後稱已獲准。未通過只阻擋該candidate，不阻擋C。artifact取得後actualSHA不符registry立即fail；不得改expected值來湊。

## 6. 公平比較與統計協議

### 6.1 輸入與分支

- 同23／13／30集合，truth來自operator，不從demo按鍵推定。
- 固定decode／YuNet／align3／quality policy。每張圖準備一次來源crop與diagnostics，candidate用相同來源，不能再呼叫DeepFace內部detect／align。
- 模型必要的resize／normalize明列，這是被比較backend的一部分，不宣稱「前處理完全一致」。
- L保留D5品質mask；R跳過gate但仍要求單一可偵測臉。離線可共享embedding、L拒絕時不公開score；這不授權live在拒絕幀上推論。
- candidate gallery缺任一身分，不能縮小gallery繼續比較；probe失敗留 `unprocessable` 一列，分母不變。

### 6.2 不直接套SFace門檻

1. SFace control固定0.363／0.3／0.10。
2. 新模型先報top1正確k／13、wrong-identity、target與non-target score/margin分布；不依門檻的ranking是共同主指標。
3. 新模型accept／FA用自己的cosine探索frontier。網格match為−1到1、step0.01，加no-accept sentinel；margin為0／0.02／0.05／0.10／0.15／0.20／0.30／0.50／1／2。完整grid留外部，報告摘frontier。
4. 新模型沒有獨立review來源時不硬塞0.3；報accept／nonaccept，review/unknown標「未配置」。官方1:1 verify threshold可引用，但不當23人open-set的現成margin／review門檻。
5. 比較本批**相同觀察FA budget**0／30、1／30、2／30下的最高correct accept與wrong identity，明寫是post-hoc描述性envelope，不是獨立校準。
6. 所有 `selected=False`。D6不選產品門檻；正式operating point要另做預先固定／獨立calibration與新資料驗證。

### 6.3 速度

分開init/model load、warm embedding、resize/normalize、IPC、cosine compare。accuracy用第一次完整評分；timing先用synthetic crop暖機一次，再對固定順序做5次warm重複。失敗不算0ms；CPU為基準，hardware/thread設定記錄。不同加速器另列，不當純模型速度比較。

## 7. Documentation impact

`docs-check: arch=N adr=N area=Y`。

既有open-source-stack已允許separately-approved非出貨benchmark；ADR0004的產品ONNX路徑不變。本次不寫「改採DeepFace」架構ADR。area是eval操作與PROJECT-STATE的研究現況。

**Implementation blocking dependency D0：** lead先執行 `project-docs-maintain`，查DeepFace／D5／選型／generation並記exact surfaces；若出現arch/ADR影響先停/query。

建議功能協議落 `docs/specs/2026-09-30-d6-deepface-comparison.md`（本plan §§1–6為內容來源）；plan落 `docs/plans/2026-09-30-d6-deepface-comparison-plan.md`。本地只寫commander `plans/`，repo落点/landing mode由lead最終確認。

結果權威分別是 `docs/research/2026-09-30-d6-deepface-comparison.md` 與 `docs/research/2026-09-30-quality-gate-measurements.md`。PROJECT-STATE只留指標/限制和pointer，不複製逐張資料。

## 8. Files與interfaces

以下是**計畫新增**，不是宣稱已存在：
- `src/facecore/eval/quality_audit.py`：C，復用diagnostics/measure。
- `src/facecore/eval/deepface_bridge.py`：subprocess embedder client，不import DeepFace。
- `src/facecore/eval/model_comparison.py`：registry、獨立gallery、L/R、frontier、timing。
- `experiments/deepface/worker.py`、`pyproject.toml`、`uv.lock`、`candidates.json`：隔離環境/metadata。
- `tests/eval/test_quality_audit.py`、`test_deepface_bridge.py`、`test_model_comparison.py`；`tests/experiments/test_deepface_worker.py`。

Consumes：既有 `ScoringContext`、`ResearchGallery`、`AlignedCrop`、`FrameDiagnostics`、`_build_true_context`、D5 `PhotoResult`／`summarize`／`summarize_r`、`write_report`。不為此改產品governance的128維常數，comparison只用研究gallery。

**Proposed bridge API：**
- `DeepFaceBridgeEmbedder.model_version: str`。
- `embed(crop: AlignedCrop) -> tuple[np.ndarray, str]`：1D、finite、nonzero、L2-normalized，dim/version匹配registry。
- `.close()`：停止並reap自己的child；先裝cleanup再launch。EOF/超時/child error不fallback SFace。

**Proposed IPC v1：** length-prefixed JSON走本機pipe。request有request_id/candidate_id、shape/dtype/color與crop base64；reply有同id、weightSHA、embedding/dim/model timing。1MiB request/reply上限，一次一個request；每request30秒timeout、init120秒timeout，失敗終止reap。payload不入一般log。

外部artifacts放 `~/Downloads/face_sample/_facecore/reports/d6-<full-head>/`：corpus manifest/source hashes、run metadata、逐張CSV、quality/timing CSV；同目錄已存在則fail-clear不覆寫。預設不存crop/embedding。migration/auth=none；rollback=停child/停研究入口，App維持SFace。

## 9. Tasks與test-first

### D0：文件與候選freeze（lead，blocking）

Files：spec/plan、`docs/research/2026-09-30-deepface-candidate-gate.md`、PROJECT-STATE pointer。

Produces：docs-check、operator確認的individual候選/用途、exact acquisition policy。未批准candidate不進load。資料未核可不阻擋C。

Acceptance：所有載入候選有exact來源、weight rights狀態/checksum策略；未解事項明列，不把wrapper MIT當weight授權。

### C：品質數值量測（與E並行）

Files：quality_audit/測試/C報告。Owner由lead指定。

Input：已有模型與23／13／30圖片。Output：66列或明確unprocessable；gallery23只做反事實診斷，不改其資格；probe13＋non-target30核對D5 mask。記mean_luma、clipped_fraction、sharpness及三個exposure predicates，保留多條同時觸發。

RED cases：mean=40/215、clip=0.05邊界；dark-only／bright-only／clip-only／多因。沿用inclusive luma區間、strict clip>0.05。測試紅在assertion，不是import error。

```bash
uv run --extra dev python -m pytest tests/eval/test_quality_audit.py -q
uv run --extra dev python -m facecore.eval.quality_audit \
  --config "$HOME/Downloads/face_sample/_facecore/g3-local.json" \
  --probes "$HOME/Downloads/face_sample/辨識組" \
  --nontarget "$HOME/Downloads/face_sample/non-target" \
  --out-dir "$HOME/Downloads/face_sample/_facecore/reports/d6-quality"
```
新增入口實作後才可跑；existing out-dir fail-clear。

Acceptance：66列完備、43列mask可核對；若加full-image曝光對照，與aligned-crop數值分開，不能套同門檻即宣布bug。可說離門檻多遠，不能因此裁定應放寬多少。

Mutation：合併三種成因、掉拒絕列、偷改原圖而非aligned crop，對應測試各RED。只讀守護針對consumer真的會讀的路徑。

### E：環境／input compatibility spike（90分鐘）

Files：隔離環境pins／candidate evidence。Owner由lead指定；未過D0不載新weights。

Questions：Mac arm64的pinned DeepFace/runtime可否解相依；skip真不再detect/align；RGB→BGR、resize、normalization是否符exact artifact；既有SFace能否同bytes橋接。

Evidence：exactPython/engine/lock、resolver結果、synthetic色塊/暗值測試、missing cache拒絕、init/退出。若已有權重資格，可測同SFace output差異；不事先假定等價容差。

Produces：具體backend interpreter與lock、worker command、model input contract。90分鐘不能證明則UNVERIFIED，保留error，不降產品Python、不改root相依。runtime版本必須由resolver 實測產生，不在plan捏造。

### A1：adapter／worker（D0＋E後，獨立PR）

Files：bridge/worker/registry/兩組測試。只做adapter，不混入統計。

RED cases：錯channel；偷偷二次detect/align；request/model/hash mismatch；zero/NaN/Inf vector；wrong dimension；missing artifact嘗試download；EOF/超時/worker崩潰/reap失敗。preprocessing測試要真的穿过worker入口，底層stub只替代model forward。

Acceptance：embed contract成立；core預設import不載重框架；child可close/reap；offline inference守住。mock pass與real weights pass分列。

Mutation：移除color convert、model/hash guard、加入推論時download、timeout後不reap，測試各RED並附失敗位置；不得留下孤兒process。

### A2：comparison runner（A1後，獨立PR）

Files：comparison/測試。復用D5可用的計數接口；新模型未配置review不得假造0.3。若修改D5入口，加選填backend且保留舊 defaults/回歸測試；其他產品路徑不動。

RED cases：新probe混舊gallery；同dim不同model；L/R用錯top1；拒絕剔分母；no runner-up卻matched；FA只看score；gallery少一人；全cohort reject；candidate共享threshold；error列當成功。

Acceptance：同raw corpus、獨立candidate digest、完整13/30分母、per-row provenance、selected=False、原D5 command不變；C mask不隨recognition模型變。

Proposed command（落地後，backend path取自E）：
```bash
uv run --extra dev python -m facecore.eval.model_comparison \
  --config "$HOME/Downloads/face_sample/_facecore/g3-local.json" \
  --probes "$HOME/Downloads/face_sample/辨識組" --probe-truth enroll-23 \
  --nontarget "$HOME/Downloads/face_sample/non-target" \
  --candidates experiments/deepface/candidates.json \
  --backend-python /absolute/path/to/approved-backend-venv/bin/python \
  --out-dir "$HOME/Downloads/face_sample/_facecore/reports/d6-comparison"
```

### A3：本機真實run＋SFace control（A2 merge後）

Consumes：已批准weights與固定corpus；66張本機處理、無相機。核corpus/store/model hashes、D5 control、C mask。

同SFace bytes bridge若不同，先查color/scale/resize/runtime，不能當模型提升。每candidate完整43列、gallery23/23、不同epoch資料不混；errors保分母；現有store/source前後hash相同。

### R：彙總報告（C＋A3後）

Files：D6/C報告、candidate gate evidence、PROJECT-STATE短pointer。

內容：各candidate L/R ranking、correct/wrong/FA、review（若配置）、nonaccept、quality、unprocessable的k/13、k/30；同FA-budget探索frontier；cold/warm cost；exact weight/input/provenance；C三因數值；小樣本與固定crop限制。repo不放逐張身分/score列表。

Reviewer由外部CSV獨立重算、核全部分母及實際CI execution，不只採作者formatter結果。

## 10. PR boundaries與verification

- PR-D：文件/候選evidence/spec/plan/docs-check；不宣稱實作完成。
- PR-C：audit＋synthetic tests；C真實報告可另docs-only PR。
- PR-A1：isolated adapter/worker，依D0/E。
- PR-A2：comparison runner，依A1。
- 本機A3：依A2與批准artifacts。
- PR-R：報告/現況，等C與A3。

review_class/branch/typed assignment由lead依live protocol處理，本plan不派review。

```bash
uv run --extra dev python -m pytest tests/eval/ tests/experiments/ -q
uv run --extra dev python -m pytest tests/ -q
uv run --extra dev --extra research-ui python -m pytest tests/ -q
uv run --extra dev ruff check src tests experiments/deepface
uv run --extra dev mypy src
```
`tests/experiments`與experiment檔案是計畫新增，落地後才跑。real backend在E凍結的環境另驗，不以mock替代。CI不得下載未批准weights／用真人照片；同head確認新測試實際passed而非skip。每個guard附mutation RED原因，不能紅在錯signature/import上。

scope evidence：root相依/live/research/profile/governance/歷史eval/D5報告無行為diff。read-only evidence：source/store前後hash相同；artifact missing/hash mismatch明確拒絕，不回落舊run。

## 11. Stop conditions與完成定義

停/query：需改產品gate/three-frame/gallery/依賴；未核weights/雲端臉API/相機；SFace control不復現；same-weight bridge無解差異；gallery23人不齊；舊資料變更；套SFacethreshold宣稱新model好壞；sweep選產品門檻；E超時仍無可行環境。

完成：control＋至少一個已核新model的同協議L/R結果、C量測、獨立復算、報告/scope evidence。候選不足交付shortage，不能稱比較完成。D6不以Qt修復、mobile移植、新增真照片digest測試或商用資格全面結案作前置。

最高風險：DeepFace內建preprocess混入模型比較；post-hocfrontier被當獨立準確率；mock/skip遮住真正入口；cache自動下載；child洩漏；只靠embedding digest誤認相同corpus。

## 12. 上游immutable evidence

- [DeepFace signature/BGR](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/deepface/DeepFace.py#L448-L473)
- [skip/resize/normalize呼叫鏈](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/deepface/modules/representation.py#L118-L203)
- [normalization與resize](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/deepface/modules/preprocessing.py#L10-L113)
- [setup相依接線](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/setup.py#L60-L66)
- [實際requirements](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/requirements.txt)
- [cache/download行為](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/deepface/commons/weight_utils.py#L23-L92)
- [SFace adapter](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/deepface/models/facial_recognition/SFace.py#L17-L79)
- [Facenet512](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/deepface/models/facial_recognition/Facenet.py#L53-L104)
- [ArcFace](https://github.com/serengil/deepface/blob/e1b38a8fced707b8acb137bcc0052302a3cae329/deepface/models/facial_recognition/ArcFace.py#L46-L95)
- [SFace目錄授權](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/README.md#L61-L63)
- [H5 mirror metadata revision](https://huggingface.co/api/models/serengil/deepface/tree/897aaade186abd2b9aeba45038a27d16c2df5f4e?recursive=false&expand=false)

截至plan完成：C未派工、individual候選未正式批准、DeepFace未安裝、新weights未下載、沒有任何DeepFace真實結果。下一步是operator檢視plan與候選建議，不是自動派工。
