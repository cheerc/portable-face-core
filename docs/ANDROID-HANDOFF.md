# Android 移植交接（portable-face-core 收尾文件）

- 日期：2026-10-09
- 上位決策：`d-20261009125609621457-5`（專案收尾、停止演算法實驗、交付本文件）、`d-20261009075738925782-1`（G3 改用 ArcFace R50，限自用）、`d-20261009080904985166-2`（ArcFace 接入實作裁定）
- 基準程式碼：本文件合併後的 `main`；之後會打 tag `android-handoff-baseline`。新 repo 請一律對這個 tag 讀碼，不要對浮動的 `main`。

這份文件寫給新的 Android repo。讀完應該能回答三件事：

1. 要從這個 repo 移植哪些東西（第 2 節）。
2. 哪些結論已經成立、不必重做實驗（第 3 節）。
3. 換到 Android 之後哪些東西一定要重新驗證（第 4 節）。

文件裡寫到的參數值、公式和程式碼位置，都是依 `07ad96f` 時點的原始碼逐行核對後寫下的。行號以上面那個 tag 為準。依本 repo 的文件慣例（`CLAUDE.md` Documentation Discipline），量測證據只給 locator（decision、報告路徑、PR），不在這裡重抄分數數字。

---

## 0. 一句話結論

在 Mac 上，**YuNet 2023mar 偵測 → 5 點相似變換對齊到 112×112 → ArcFace R50（`w600k_r50.onnx`）512 維 embedding → cosine 比對 → 連續 3 幀支持才判 matched** 這條流程已確認可用：

- operator 用 G3 App 做過家人實測，接受結果（`d-20261009125609621457-5`）。
- 實作已合併（PR #171）。

Android 端只要照搬**同一組 ONNX 模型與同一套前處理**，選型結論就能沿用。要重做的只有三件事（詳見第 4 節）：**逐階段數值等價驗證、門檻與品質關卡重校、手機效能量測**。

---

## 1. 這個 repo 確認了什麼、沒確認什麼

| 已確認 | locator |
|---|---|
| 換成 ArcFace R50 後，SFace 時期「本人與家人分數重疊」的問題消失，operator 實測接受 | `d-20261009075738925782-1`、`d-20261009125609621457-5` |
| repo 內的 ArcFace 實作與 operator 實測用的 reference 數值等價（逐張 embedding 在 1e-6 內） | PR #171；`docs/research/2026-10-09-arcface-r50-license-provenance.md` |
| 模型完整性：SHA 不符或檔案缺失就拒絕啟動，沒有 fallback | `src/facecore/pipeline/embed.py:27-42`、`src/facecore/research/cli.py:1072-1090` |

| **沒有**確認（不要當成已知） | 說明 |
|---|---|
| 代表性人群的準確率、低 FAR | 只有一個家庭的小樣本 |
| liveness／防照片翻拍 | 完全沒有做；`matched` 不等於「已驗證本人」 |
| 手機上的延遲、耗電、發熱 | 只跑過 ORT mobile checker 的靜態分析 |
| 上游 InsightFace `norm_crop` 與本 repo 對齊結果的數值等價 | 見 ADR 0011「已知限制」 |

---

## 2. 移植清單（依資料流順序）

### 2.1 模型檔與完整性檢查

| 模型 | 檔名 | SHA-256 | 授權 | 來源 |
|---|---|---|---|---|
| 偵測 YuNet 2023mar | `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT（opencv_zoo，見 `docs/research/2026-09-10-model-candidate-gate.md` §Pair 1） | `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` |
| 辨識 ArcFace R50 | `w600k_r50.onnx`（取自 `buffalo_l.zip`） | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` | **僅限非商業研究**（原文：「ALL models are available for non-commercial research purposes only」）；本專案依 decision 定性為自用、非商用、不散佈 | `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip` |

- **要移植的設計**：載入模型前先算 SHA-256，與寫死的期望值比對，不符就拒絕啟動（`embed.py:27-42`、`yunet.py:143-157`）。**不准 fallback 到其他模型**：repo 外那支早期 reference runner 在缺 R50 時會靜默改用 MBF，PR #171 已改成直接拒絕啟動。
- **權重不進 Git**（repo 是 public，而且 ArcFace 授權禁止散佈）。Android 自用的做法：把模型 `adb push` 到 App 私有目錄，或在本機打包時放進不入 Git 的 assets；載入時一律驗 SHA。
- 若之後要商用或公開發佈 App，必須先取得 InsightFace 商用授權（見 ADR 0011「授權與 provenance」）。

### 2.2 影像解碼與方向

- 相機影格在推論前要先依旋轉角轉正（`frame_pipeline.py:376-385`：`np.rot90`，k = orientation/90）。
- **鏡像只影響預覽顯示，推論一律用原始方向的影像**（`frame_pipeline.py:378-379` 註解）。前鏡頭預覽通常是鏡像，切勿把鏡像後的像素送進模型。
- 註冊照片：見 2.8 的 EXIF 陷阱。

### 2.3 偵測：YuNet（`src/facecore/pipeline/yunet.py`）

| 項目 | 值 | 位置 |
|---|---|---|
| 輸入 | 2023mar 只接受**固定 640×640** | `yunet.py:12-16, 30` |
| 縮放 | 等比縮放（單一 scale）＋**置中**補零 letterbox，bilinear | `yunet.py:110-130`；ADR 0009 Decision 1 |
| 再補齊 | 長寬補到 32 的倍數（edge 模式；640 本來就整除） | `yunet.py:133-143` |
| 色彩／數值 | **BGR**、raw [0,255]、NCHW float32，不做 normalize | `yunet.py:10-11, 174-180` |
| IO 名稱 | 輸入 `input`；輸出每個 stride 各一組 `cls_{s}`／`obj_{s}`／`bbox_{s}`／`kps_{s}`，s ∈ {8,16,32} | `yunet.py:181-203` |
| 解碼 | `score = sqrt(clamp(cls)·clamp(obj))`；`cx=(c+dx)·s`、`cy=(r+dy)·s`、`w=exp(dw)·s`、`h=exp(dh)·s`；landmark `(kps+c)·s, (kps+r)·s` | `yunet.py:41-82` |
| 門檻 | `score_threshold=0.9`、NMS IoU `0.3`、`top_k=5000` | `yunet.py:163-172` |
| 座標還原 | `(x − pad) / scale` 換回原圖像素 | `yunet.py:205-220` |
| 單臉規則 | 0 張臉 → 該幀拒絕；**多於 1 張 → 整個 session 判 `invalid_input`，需重新開始** | `pipeline/detect.py:13`；`live/session.py` 多臉分支 |

> 解碼邏輯是照 OpenCV `face_detect.cpp:174-245` 移植的。Android 若改用 OpenCV `FaceDetectorYN`，理論上行為相同，但仍要用 golden 測資比對（見第 4 節）。

### 2.4 對齊：5 點相似變換（`src/facecore/pipeline/align.py`，contract v3）

- 輸出：112×112 RGB。
- 標準 template（取自 OpenCV `face_recognize.cpp`；與 InsightFace `arcface_dst` 相同），`align.py:28-39`：
  ```
  (38.2946, 51.6963)  右眼
  (73.5318, 51.5014)  左眼
  (56.0252, 71.7366)  鼻尖
  (41.5493, 92.3655)  右嘴角
  (70.7299, 92.2041)  左嘴角
  ```
  landmark 順序與 YuNet 輸出順序一致，**不重排**。
- 變換：Umeyama 相似變換（旋轉＋等比縮放＋平移，SVD，處理反射），`align.py:43-80`。
- 重取樣：**bilinear**。
- **邊界：先把原圖四周以 edge 模式補 `max(1, round(max(w,h)·0.25))` 像素再 warp**（`align.py:103-110`）。邊界不可以填 0，否則黑角會觸發曝光品質關卡。
- ⚠ InsightFace 官方 `norm_crop` 用的是 `cv2.warpAffine(borderValue=0)`，與本 repo 不同。operator 實測接受的是**本 repo 的對齊**。Android 端要以本 repo 的輸出為準：OpenCV `warpAffine` 搭配 `INTER_LINEAR` ＋ `BORDER_REPLICATE` 最接近，但仍要用 golden 測資驗證。

### 2.5 品質關卡（`src/facecore/pipeline/quality.py`、`measure.py`、`contracts/policy.py:32-44`）

`PolicyProfile.frozen_v1()` 的七道關卡，任一不過該幀就拒絕（多個失敗會一起回報）：

| 關卡 | 值 | 量測方式 |
|---|---|---|
| 偵測信心 | ≥ 0.90 | YuNet score |
| 臉的短邊 | ≥ 112 px | 偵測框 `min(w,h)`（原圖像素） |
| 清晰度 | ≥ 60.0 | 對齊後 luma 的 Laplacian 變異數（4 鄰域核） |
| 曝光 | mean luma ∈ [40, 215]，且 clipped 比例 ≤ 0.05 | luma = 0.299R+0.587G+0.114B；clipped = luma ≤2 或 ≥253 |
| 偏航 yaw | ≤ 30° | `|nose_x − 兩眼中點x| / 兩眼距 × 90` |
| 俯仰 pitch | ≤ 20° | `|nose_y − 眼嘴中線y| / 眼嘴垂直距 × 90` |
| 遮擋 | 0 個低信心 landmark | YuNet 沒有逐點信心，固定為 1.0，所以實際上不會觸發 |

yaw／pitch 是**粗略代理值**，不是真實角度（`measure.py:10-12` 自陳）。

### 2.6 Embedding：ArcFace R50（`src/facecore/pipeline/embed.py`、`contracts/manifest.py:49-90`）

| 項目 | 值 |
|---|---|
| 輸入 | 對齊後 112×112 **RGB**（不是 BGR） |
| 前處理 | `(pixel − 127.5) / 127.5`，NCHW float32 |
| IO 名稱 | 輸入 `input.1` `[N,3,112,112]`；輸出 `683` `[N,512]`。名稱要從 session 讀取，不要寫死 |
| 後處理 | **L2 normalize**；norm 為 0 就報錯 |
| model version | `arcface-w600k-r50-fp32` |

### 2.7 比對與單幀判定

- 分數：L2 normalize 後的 cosine（內積）。
- 每幀把 probe 與 gallery 每個身分比對，取 top1、top2；`margin = top1 − top2`。
- 單幀是否「支持」top1：`top1 ≥ match_threshold` **且** `margin ≥ margin_threshold`（`live/session.py` observe 段）。gallery 只有一個身分（沒有 runner-up）時，該幀不算支持。
- **跨模型拒絕比對**：probe 與 gallery 的 model version 不同時直接報錯，不給分數（`embed.py:64-77`）。換模型或量化版本時，gallery 必須重建。

### 2.8 Session（連續幀判定，`src/facecore/live/session.py`；參數在 `profiles/g3-v1-arcface-r50.json`）

| 參數 | 值 | 意義 |
|---|---|---|
| `match_threshold` | **0.60** | 單幀支持門檻（ArcFace 語境；SFace 的 0.363 不可沿用） |
| `margin_threshold` | 0.10 | top1 − top2 下限 |
| `review_threshold` | 0.30 | 逾時時用來把最佳幀歸成 review 或 unknown 的診斷分界 |
| `required_support` | 3 | 同一身分累積 3 個支持幀即判 `matched` |
| `min_support_interval_ms` | 200 | 兩個支持幀至少間隔 200ms，間隔不足的幀跳過（不清空） |
| `sample_interval_ms` | 200 | 取樣間隔 |
| `timeout_ms` | 5000 | 辨識窗口 |
| `max_frames` | 26 | 窗口內最多處理的幀數 |
| `queue_limit` | 1 | 只保留最新一幀（推論跟不上就丟舊幀） |
| `continuity_max_center_delta_ratio` | 0.5 | 臉中心位移 ÷ 前一幀臉尺寸 > 0.5 → `invalid_input`（防換人） |

判定規則：

- **任一**不支持事件（沒有臉、品質不過、分數或 margin 不足、top1 換人）都會**清空**支持窗口，重新累積。
- 窗口從**第一個有效影格**開始計時（`session.py:273-303` `anchor_recognition`），不從按下開始。
- 逾時：完全沒有可用幀 → `invalid_input`，並細分為 `no_frames_captured`／`all_frames_rejected_no_face`／`all_frames_rejected_quality`／`all_frames_rejected_mixed_causes`；有可用幀但湊不滿支持 → `timeout`，附最佳幀落在 review 或 unknown。

### 2.9 註冊（gallery）

- 規則（`frame_pipeline.py:166-289`）：一張照片代表一個身分，身分名＝檔名 stem；只收**恰好一張臉**的照片；同名重複就拒絕；全部失敗則整個 gallery 建立失敗。
- 每次 App 啟動都從照片重新建立 in-memory gallery；沒有持久化 template。
- 註冊照片**沒有經過品質關卡**，只檢查單臉。
- Android 端要做持久化時：embedding 屬於生物特徵資料，要加密保存，並帶上 model version 欄位，換模型時強制重建（呼應 2.7）。

### 2.10 值得移植的防護設計

1. 模型 SHA 檢查（2.1）。
2. **模型與 profile 配對檢查**：ArcFace embedding 只能搭配 ArcFace profile（門檻 0.60）；用實際的 model version 判斷，不靠檔名；未知的 model version 一律拒絕（`research/cli.py:1072-1085`、PR #171）。
3. 跨模型拒絕比對（2.7）。
4. 多臉時整個 session 失效（2.3）。

---

## 3. 不必重做的結論（附依據）

| 結論 | 依據 |
|---|---|
| **不要再評估 SFace**：本人與未註冊家人的分數範圍重疊，任何單一門檻都分不開 | `d-20261009075738925782-1`（依據含 operator 本機的 2026-10-08 離線評估，該報告在 repo 外、不入 Git） |
| **不要再評估 Facenet512**：分數整體偏高，SFace 的門檻不能套用，也不比 ArcFace 好 | `docs/research/2026-09-30-d6-deepface-comparison.md`；PR #136 |
| deepface 的 Keras ArcFace 跑不起來，與 InsightFace ONNX 版無關，不必追 | 同上 §6.3 |
| **MBF（`w600k_mbf.onnx`，13MB）的區分力明顯弱於 R50**，不宜直接當預設 | `d-20261009075738925782-1` 第 5 點 |
| 偵測輸入必須等比縮放＋置中補零；非等比縮放會傷害分數 | ADR 0009 |
| 對齊要用 5 點相似變換，不能只裁框再縮放 | ADR 0009 Decision 2；`align.py` contract v3 |
| 計時從第一個有效影格開始；把開相機的時間算進窗口會造成大量 `invalid_input` | `docs/mac-demo-baseline-d0.md`；D1（PR #119–#125） |
| D5 靜態照片集（SFace 時期）裡，品質拒絕全部來自**曝光關卡**；換鏡頭時最先要看曝光拒絕率，而不是先懷疑模型 | `docs/research/2026-09-30-d5-static-baseline.md`；`docs/PROJECT-STATE.md`「D5 現況」 |
| 授權審查（YuNet MIT；ArcFace 限非商業研究） | `docs/research/2026-10-09-arcface-r50-license-provenance.md`；ADR 0011 |
| ORT mobile checker：YuNet 可整張交給 NNAPI；R50 必須**固定輸入 shape** 才能交給 NNAPI／CoreML NN | `docs/research/2026-09-30-d6-m-onnx-mobile-usability.md`；`manifest.py:76-90` |

---

## 4. 換到 Android 一定要重新驗證的

1. **Golden 測資的數值等價（第一優先）**：用本 repo 的程式，對合成或已取得同意的影像產生「偵測框＋landmark → 對齊後 112×112 像素 → embedding」的 golden 檔，再讓 Android 端逐階段比對。對齊階段的容差要自訂（像素誤差）；embedding 的 cosine 參考 PR #171 用的 1e-6。不先做這一步，第 3 節的結論就不能沿用。
2. **門檻重校**：手機相機的畫質、曝光、對焦，以及對齊實作的差異都會讓分數分布移動。0.60 只能當**起始值**，要在 Android 上用家人實測重新確認。已知 0.60 對其中一位成員已接近通過下限，見 `d-20261009075738925782-1` 的實測 provenance。
3. **品質關卡的值**：曝光 [40,215]、清晰度 60 這些值是在 Mac 鏡頭上凍結的，手機鏡頭需要重新觀察拒絕原因的分布。
4. **效能**：每幀 detect + align + embed 最好在 200ms（取樣間隔）內完成；R50 FP32 在中階手機上的延遲**未知**，要實測，並分別記錄冷啟動、穩態、長時間發熱降頻三種情況。
5. **若做了量化（FP16／INT8）或改用 MBF**：等同換模型，第 1、2 項都要重做，gallery 也要重建。

---

## 5. 已知陷阱

- **註冊照的 EXIF 方向**：本 repo 的 G3 註冊路徑用 `Image.open(...).convert("RGB")`，**沒有套用 EXIF 方向**（`frame_pipeline.py:210-211, 324-325`），但另一條 decode 路徑有套用（`pipeline/decode.py:20-23`）。手機拍的直式照片幾乎都帶 EXIF 旋轉，Android 端註冊時**務必先依 EXIF 轉正**。
- **YuNet 2023mar 只接受 640×640**，其他 shape 會被拒絕（`yunet.py:12`）。
- **ArcFace ONNX 的 IO 名稱不是語意名**（`input.1`／`683`），要從 session 讀取。
- **ArcFace 要 RGB、YuNet 要 BGR**，兩者不同，最容易接錯。
- **NNAPI 已被 Google 從 Android 15 起 deprecate**，ONNX Runtime 也已標記其 NNAPI EP 為 deprecated（見第 6 節來源），不要以它為主路徑。
- R50 原始模型是 dynamic shape；要用 NPU 或 GPU 類 EP，通常得先輸出固定 batch=1 的版本。改了 graph 就要重驗 golden 測資。
- w600k ONNX 有 initializer 出現在 graph inputs（ORT 會警告），影響部分圖最佳化；可用 ORT 官方工具 `remove_initializer_from_input.py` 清除，但這也算改模型，要重驗。
- 單幀分數過門檻不等於 matched，必須連續 3 個支持幀；中途任何拒絕都會歸零。這是刻意設計（寧可錯殺，不可錯放；`d-20261003082352062664-3`）。

---

## 6. 建議的 Android 技術方案

**原則：沿用同一份 ONNX 模型和同一套前處理，讓第 3 節的結論能直接沿用；只有實測證明效能不夠時才換推論框架。**

| 層 | 建議 | 理由 |
|---|---|---|
| 語言 | Kotlin | Android 官方主語言 |
| 相機 | **CameraX `ImageAnalysis`**，背壓策略 `STRATEGY_KEEP_ONLY_LATEST` | 對應 `queue_limit=1`（只處理最新一幀）；用 `ImageInfo.rotationDegrees` 轉正（2.2） |
| 推論 | **ONNX Runtime Android**（Maven `com.microsoft.onnxruntime:onnxruntime-android`），YuNet 與 ArcFace 都用它 | 直接吃現有 `.onnx`，不必轉檔，數值最容易對齊。ORT 行動部署指南建議：非量化模型先用 **XNNPACK EP**，量化模型先用 CPU EP；預建的 Android 套件已內含 XNNPACK |
| 加速（選用） | 實測不夠快時：Snapdragon 機型可試 **QNN EP**（需要固定 shape，而且要 profile 確認真的跑在 NPU 上）；跨廠牌的 GPU 則要改走 **LiteRT**（`.onnx`→`.tflite` 轉檔），轉完必須重驗 golden 測資 | NNAPI 已 deprecate；ORT 目前在 Android 上沒有通用的 GPU EP |
| 對齊／影像處理 | Kotlin 移植 `similarity_transform`（約 40 行），warp 用 OpenCV Android `warpAffine(INTER_LINEAR, BORDER_REPLICATE)`，或自寫 bilinear 加 edge padding | 要對齊本 repo 的 align3（2.4），用 golden 測資驗收 |
| 偵測（替代方案） | 也可以用 OpenCV `FaceDetectorYN` 載入同一份 YuNet | 少寫解碼程式，但要驗證門檻與 NMS 參數一致（0.9／0.3） |
| **不建議** | ML Kit Face Detection、MediaPipe Face Detector 取代 YuNet | landmark 定義與框不同，對齊結果會改變，第 3 節結論就不能沿用。ML Kit 本身也不做身分辨識 |
| 資料保存 | embedding 用 Android Keystore 產生的 AES-GCM 金鑰自行加密後存檔或存 DB，記錄 model version | 生物特徵資料（`CLAUDE.md` Design Principles） |
| 模型散佈 | 自用：`adb push` 到 App 私有目錄，或本機打包時放進不入 Git 的 assets；載入時驗 SHA | ArcFace 授權禁止散佈；public repo 不放權重 |

版本號請在新 repo 開工當下釘選當時的穩定版，本文件刻意不寫死。

來源（2026-10-09 查閱）：
- ONNX Runtime 行動部署指南（EP 選擇建議）：https://onnxruntime.ai/docs/tutorials/mobile/
- ONNX Runtime XNNPACK EP：https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html
- ONNX Runtime QNN EP：https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- ONNX Runtime NNAPI EP 頁與 deprecate 討論：https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html 、https://github.com/microsoft/onnxruntime/issues/23206
- Android NNAPI（Android 15 起 deprecated）與遷移指南：https://developer.android.com/ndk/guides/neuralnetworks 、https://developer.android.com/ndk/guides/neuralnetworks/migration-guide

---

## 7. 新 repo 的起手順序

1. 讀本文件，以及 ADR 0009、ADR 0011、`profiles/g3-v1-arcface-r50.json`。
2. 用本 repo（tag `android-handoff-baseline`）產生 golden 測資，只能用合成或已取得同意的影像，不入 public Git。
3. Android 端做出最小管線：CameraX → YuNet → align → ArcFace，逐階段對 golden 測資。
4. 實作 2.8 的 session 規則與 2.10 的防護。
5. 在手機上量延遲；不夠快才進入第 6 節的加速選項。
6. 用家人實測重校門檻與品質關卡值（第 4 節第 2、3 項）。

---

## 8. 不需要移植的

這些是 Mac 研究階段的工具或治理機制，Android App 不需要：

- Qt 介面（`live/qt_window.py`、`controller.py`）與 research CLI（`research/cli.py` 的 live／replay 子命令）。
- 39 欄 demo CSV 紀錄（`G3_DEMO_RESULTS_CSV_COLUMNS`）。除錯時可參考它的欄位設計，但不必照搬。
- Phase-1A／1B 的 governance、lifecycle、shadow-candidate 模板累積、drift 監控（`governance/`、`policy/identify.py`），以及 D5／D6 評估工具（`eval/`）。若 Android 之後要做「自動累積新模板」，先讀 ADR 0003、0005 的設計再決定。
- SFace 相關的 manifest 與 profile（`g3-v1.json`）。
