# D6：DeepFace 模型比較（功能協議）

- 日期：2026-09-30
- 狀態：**協議已定義，尚未實作。** 截至本檔落地時，D6 的 PR-D／C／E 是 plan §9 設計的 prep 階段，
  A1／A2／A3／R 未開工，`src/facecore/eval/` 下尚無 D6 模組，DeepFace 未安裝，沒有任何 DeepFace 真實結果。
- 修訂基準：main `592493968eb5f47bbd37b57d4e1f275330e7a56c`（D5 PR #127）。
- 內容來源：[D6 計畫](../plans/2026-09-30-d6-deepface-comparison-plan.md) §§1–6。本檔不複述 plan 的
  授權邊界與 stop conditions，兩者關係見 §7。
- 範圍裁決：`d-20260930132146694094-31`（D6 parent `t-20260930131754483952-84237-116`）。

## 1. 目標

用相同 23 張註冊照、13 張辨識照、30 張未註冊照與明確前處理，比較現有 SFace 與 DeepFace 候選的
辨識排序、正確接受、錯誤接受及成本；同時量出品質拒絕的數值原因。**不改現有 App。**

採用**固定 detector／landmark alignment，只換辨識 backend**，不把偵測、對齊、品質 gate、模型、門檻
全部換掉後統稱為「演算法差異」。

| 方案 | 本計畫處理 |
|---|---|
| 固定 YuNet＋align3，換 embedder | 採用；主要比較模型及其必要前處理 |
| DeepFace 預設完整 pipeline | 不做；它同時改 detector／alignment／threshold，無法隔離差異 |
| 直接替換 App 模型 | 不做；比較不是 runtime migration 或 mobile 交付 |

共同來源 crop 是既有 `align_crop` 輸出的 112×112 RGB（`pipeline/align.py`）。不同模型需要不同尺寸時，
允許 adapter 做必要 resize，但**必須記錄，不能宣稱前處理完全相同**。固定 112 crop 放大到 Facenet512 的
160 輸入是實驗限制；本結果不代表 DeepFace 原生完整 pipeline 的最佳效果。

C 的品質診斷與 A 的候選／runtime 查證可並行；正式比較報告必須引用兩邊 evidence。C 不改任何行為，
A 不回寫產品門檻。

## 2. 比較設計的事實前提

- 註冊路徑不執行品質 gate、probe 路徑執行，是已查證的程式事實；**不對稱本身不等於 bug**。同來源照片
  品質可用是 operator 的觀察，應保留；但同來源不單獨證明每張照片的 luma 相同，也不直接證明 gate 錯誤。
- D5 的 `quality_exposure` 可能來自平均 luma 低於 40、高於 215，或 clipped fraction 大於 0.05。
  **不能將 23 張全部說成「太暗」**（D5 報告 §5、`pipeline/quality.py:41-46`）。C 要分開三種條件，
  允許同時觸發。
- L2 normalization／cosine 不會讓不同模型的 score 分布相同。新模型**不能**直接套用 SFace 的
  match 0.363／review 0.3／margin 0.10。
- DeepFace 是框架，不是單一模型。不能把 InsightFace 某個權重的限制套到 DeepFace 同名 ArcFace，也不能把
  現有 SFace 一概寫成「僅非商用」。本 repo 的 SFace candidate gate 記錄 Apache-2.0 目錄聲明、commercial
  grant 與未解訓練 provenance（`docs/research/2026-09-10-model-candidate-gate.md` §1B）。
- 高相似度不證明「撞臉」，quality rejected 不證明照片有問題，R 正確不證明整體演算法無問題。報告只陳述
  本批資料的分布，不冒充成因診斷。

## 3. 接入契約

### 3.1 本 repo（base `5924939`）

| 項目 | 契約 |
|---|---|
| Embedder | `pipeline/embed.py`：`model_version: str`；`embed(crop: AlignedCrop) -> tuple[np.ndarray, str]`；SFace 輸出 L2-normalized |
| 注入 | `research/cli.py` 的 `_build_true_context(..., embedder_factory=...)`；D5 `eval/static_baseline.py` 目前不傳 factory |
| Gallery | `live/frame_pipeline.py`：immutable embeddings；digest hash sorted identities＋vector bytes，故需另記 model ID／weight hash／generation |
| 模型隔離 | `ScoringContext.__post_init__` 及 `score_frame` 的 version check |
| 比對 | `policy/identify.py` 的 `cosine_score`；不能只因兩個模型都 512 維就允許混比 |
| L／R | `static_baseline.py` 的 `score_photo`；L 呼叫 `score_frame`，R `_r_branch` 跳過 quality |
| Band | `static_baseline.py`：score ≥ match **且** margin ≥ margin 才 matched；否則 review／unknown；無 runner-up 不 matched |
| 診斷 | `live/contracts.py` 的 `FrameDiagnostics`；`frame_pipeline.py` 發布 luma／clipped／sharpness／timing |
| 量測 | `pipeline/measure.py` 的 `exposure_of`、`sharpness_of`、`pose_of`；C 直接復用，不另寫近似算法 |
| 相依 | root 要求 Python `==3.14.*`，沒有 DeepFace／TensorFlow／PyTorch |

D5 control 必須重新核對：gallery 23／23、digest `e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f`；
YuNet SHA `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`；SFace SHA
`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`；g3-v1、align3、detector gate 0.90。

預期對照：L probe correct 12／13、wrong 0、quality 1；L non-target FA 2／30、review 6、quality 22；
R probe correct 13／13、wrong 0；R non-target FA 2／30、review 28。**若同版本、資料與參數卻不同，
先查 divergence，不以舊 CSV 冒充本次 control。**

> 註：`static_baseline.py` 目前沒有任何 gallery digest 守護（repo-wide grep 確認 `tests/` 裡唯一的
> `e3d77c4c` 是無關 fixture）。D0 凍結值目前只是觀測值，D6 的 control 步驟必須人工核對，不能假設會被自動擋。

### 3.2 DeepFace（固定 `e1b38a8fced707b8acb137bcc0052302a3cae329`）

- `DeepFace.py:448-473`：ndarray 的 API 契約是 **BGR**；`represent` 有 `detector_backend`、`align`、
  `normalization`、`l2_normalize`。
- `modules/representation.py:118-203`：`skip` 不呼叫 detector／alignment，但仍 resize／normalize；
  **skip 與共同分支各反轉一次通道**，故 adapter 應先把來源 **RGB 轉 BGR**。`align=False` 不代表禁止 resize。
- `modules/preprocessing.py:10-113`：`base` 不另外做模型 normalization；`Facenet` 是 per-image mean/std；
  `Facenet2018` 是 127.5 縮放；`ArcFace` 是 `(x-127.5)/128`。**不會按 model_name 自動選。** resize 後先有
  `[0,1]` 尺度，max≤1 的輸入另有行為，需用合成暗色塊測邊界。
- 尺寸／維度：SFace 112／128 維；Facenet512 160／512 維；ArcFace 112／512 維。H5 與 PyTorch `.pth` 是不同
  artifacts，不能只記模型名。
- `setup.py:60-66` 實際讀 `requirements.txt`，其中仍包含 TensorFlow／Keras。`[pytorch]` extra **不會**
  刪除 base TensorFlow 相依。不能沿用「只裝 PyTorch 就沒有 TensorFlow」的說法。
- `commons/weight_utils.py:23-92` 只看 cache 是否存在，**沒有 checksum／offline 開關**；
  `folder_utils.py:7-34` 由 `DEEPFACE_HOME` 決定 `.deepface/weights`，import 也會建目錄。adapter 需在
  import 前設定隔離 home 並驗 cache，再阻擋推論時網路。

比較正規化暫定 SFace bridge＝`base`。**這是待查證的模型輸入契約，不是調參候選**：E 必須找到 exact
weight 對應訓練／reference 推論依據，不能因名字相近就採用。不一致則停下回報，不掃三種 normalization
挑最好的一種。

## 4. 公平比較與統計協議

### 4.1 輸入與分支

- 同 23／13／30 集合，truth 來自 operator，不從 demo 按鍵推定。
- 固定 decode／YuNet／align3／quality policy。每張圖準備一次來源 crop 與 diagnostics，candidate 用相同
  來源，**不能再呼叫 DeepFace 內部 detect／align**。
- 模型必要的 resize／normalize 明列，這是被比較 backend 的一部分，不宣稱「前處理完全一致」。
- L 保留 D5 品質 mask；R 跳過 gate 但仍要求單一可偵測臉。離線可共享 embedding、L 拒絕時不公開 score；
  **這不授權 live 在拒絕幀上推論**。
- 每個 backend 重建自己的 23 人 gallery，禁止新模型 probe 與 SFace 向量混比。原始身分／檔案 hash 相同，
  embedding digest 則按模型另外記錄；**新模型不得要求等於 SFace digest**。
- candidate gallery 缺任一身分，**不能縮小 gallery 繼續比較**；probe 失敗留 `unprocessable` 一列，分母不變。

### 4.2 不直接套 SFace 門檻

1. SFace control 固定 0.363／0.3／0.10。
2. 新模型先報 top1 正確 k／13、wrong-identity、target 與 non-target score／margin 分布；不依門檻的 ranking
   是共同主指標。
3. 新模型 accept／FA 用自己的 cosine 探索 frontier。網格 match 為 −1 到 1、step 0.01，加 no-accept sentinel；
   margin 為 0／0.02／0.05／0.10／0.15／0.20／0.30／0.50／1／2。完整 grid 留外部，報告摘 frontier。
4. 新模型沒有獨立 review 來源時**不硬塞 0.3**；報 accept／nonaccept，review／unknown 標「未配置」。
   官方 1:1 verify threshold 可引用，但不當 23 人 open-set 的現成 margin／review 門檻。
5. 比較本批**相同觀察 FA budget** 0／30、1／30、2／30 下的最高 correct accept 與 wrong identity，
   明寫是 post-hoc 描述性 envelope，不是獨立校準。
6. 所有 `selected=False`。**D6 不選產品門檻**；正式 operating point 要另做預先固定／獨立 calibration
   與新資料驗證。

### 4.3 速度

分開 init／model load、warm embedding、resize／normalize、IPC、cosine compare。accuracy 用第一次完整評分；
timing 先用 synthetic crop 暖機一次，再對固定順序做 5 次 warm 重複。**失敗不算 0 ms**；CPU 為基準，
hardware／thread 設定記錄。不同加速器另列，不當純模型速度比較。

## 5. 檔案與介面（計畫新增，尚未存在）

- `src/facecore/eval/quality_audit.py`：C，復用 diagnostics／measure。
- `src/facecore/eval/deepface_bridge.py`：subprocess embedder client，**不 import DeepFace**。
- `src/facecore/eval/model_comparison.py`：registry、獨立 gallery、L／R、frontier、timing。
- `experiments/deepface/worker.py`、`pyproject.toml`、`uv.lock`、`candidates.json`：隔離環境／metadata。
- `tests/eval/test_quality_audit.py`、`test_deepface_bridge.py`、`test_model_comparison.py`；
  `tests/experiments/test_deepface_worker.py`。

Consumes：既有 `ScoringContext`、`ResearchGallery`、`AlignedCrop`、`FrameDiagnostics`、`_build_true_context`、
D5 `PhotoResult`／`summarize`／`summarize_r`、`write_report`。**不為此改產品 governance 的 128 維常數**，
comparison 只用研究 gallery。

**Bridge API（proposed）：**

- `DeepFaceBridgeEmbedder.model_version: str`。
- `embed(crop: AlignedCrop) -> tuple[np.ndarray, str]`：1D、finite、nonzero、L2-normalized，dim／version
  匹配 registry。
- `.close()`：停止並 reap 自己的 child；**先裝 cleanup 再 launch**。EOF／超時／child error **不 fallback SFace**。

**IPC v1（proposed）：** length-prefixed JSON 走本機 pipe。request 有 request_id／candidate_id、shape／dtype／color
與 crop base64；reply 有同 id、weight SHA、embedding／dim／model timing。1 MiB request／reply 上限，一次一個
request；每 request 30 秒 timeout、init 120 秒 timeout，失敗終止 reap。**payload 不入一般 log。**

外部 artifacts 放 `~/Downloads/face_sample/_facecore/reports/d6-<full-head>/`：corpus manifest／source hashes、
run metadata、逐張 CSV、quality／timing CSV；**同目錄已存在則 fail-clear 不覆寫**。預設不存 crop／embedding。
migration／auth＝none；rollback＝停 child／停研究入口，App 維持 SFace。

## 6. 候選資格（`d-20260930132146694094-31`）

本輪可用候選只有兩個，**兩者都不引入新權重**：

| Candidate | 狀態 |
|---|---|
| `ort-sface-control` | 可用 —— 既有凍結 artifact |
| `deepface-sface-bridge` | 可用 —— 同一份已核 SFace weight bytes；價值在抓色彩／尺度／前處理接錯 |
| `deepface-facenet512` | **不可用** —— operator 逐一核准 exact weight 的 code license／weight license／provenance 前，不得下載、載入、執行 |
| `deepface-arcface` | **不可用** —— 同上 |

A1／A2／A3／R 全部停在 PR-C 與 E 之後，等 operator 裁決。候選細節（locator、SHA 來源、license、provenance
狀態）見 [D6 候選 gate](../research/2026-09-30-deepface-candidate-gate.md)。

若 bridge 顯示同權重下差異，**先查前處理接線**（BGR、resize 尺度、normalization、float→uint8），不得直接
歸因模型或 wrapper。

## 7. 本檔與 plan 的關係

- 本檔是**功能協議**（做什麼、怎麼算公平、介面契約）；plan 是**執行計畫**（授權邊界、PR 切分、test-first、
  mutation、stop conditions、完成定義）。衝突時以 plan 為準。
- plan §11「候選不足交付 shortage」是**最終**完成定義，不是本輪結論。當前的 PR-D／C／E 是 plan §9 設計的
  prep，不是 shortfall 結案。
- 本檔描述**目前設計狀態**；實作進度看 `docs/PROJECT-STATE.md` 的 pointer，授權與未解事項看
  candidate gate，兩者不重複。
