# D6 R: DeepFace 候選模型比較 — Facenet512 vs SFace control

**Date:** 2026-10-01 · **Tasks:** D6 A3 (`t-20261001025052122296-75458-16`) ·
**HEAD:** `ac695befb042f7301320fa55ea5b1d8c15c9dc45` ·
**Plan:** [`2026-09-30-d6-deepface-comparison-plan.md`](../plans/2026-09-30-d6-deepface-comparison-plan.md) §9 A3 (`:224-228`)

## 1. 範圍與授權

- **corpus 66 張**（註冊 23／probe 13／non-target 30），**比較 43 列**
  （probe 13 ＋ non-target 30），**gallery 23/23**。66 是處理量，43 是比較量，
  兩者不可混寫。
- 全部本機處理，**無相機**。未呼叫 `DeepFace.analyze`，未連雲端人臉 API。
- 授權依據 `d-20260930132705762549-32`（d-32）：僅本機研究比較，不散布、
  不商用、不進產品 runtime。**權重檔本身未進 Git**，僅 checksum 進 repo。
- **ArcFace 不在本輪執行**（理由見 §6.3）。
- **兩個 H5 的 weight license 與 training-data provenance 仍未證實
  （`UNPROVEN` / `UNRESOLVED`）。核准使用不等於 license 已解決。**

## 2. 方法

**兩臂吃同一份來源 crop**（plan §6.1）：detect／align／quality 全部是產品的
（YuNet 2023mar ＋ `align_crop` ＋ g3-v1 policy），候選**不得**呼叫 DeepFace 的
detect 或 align。crop manifest 記每組**聚合 digest ＋ 張數**，不記逐張檔名。

- **control**：`facecore.pipeline.embed.Embedder`，onnxruntime **1.30.0**，
  CPython **3.14**。
- **candidate**：`Facenet512`，DeepFace **0.0.93**，CPython **3.13.15**。
  經真 subprocess（`subprocess.Popen` ＋ fd pipe）與 control 進程分離。
- **模型必要的輸入尺寸差異**：產品 crop 為 112×112 canonical template；
  **Facenet512 宣告 160×160**，由 worker resample 一次。依 plan §6.1，
  這是被比較 backend 的一部分，明列於此。尺寸由 model 自身 layer 讀出，
  非硬編碼。
- **絕不套用 SFace 門檻**（plan §6.2）。g3-v1 的 0.363／0.3／0.10 只用於
  control 自身，候選只報不依賴門檻的 ranking 與分佈。

## 3. D5 control：精確復現

43 列、**19 個非計時欄位與記錄的 D5 CSV 逐格零差異**。唯一變動是 `stage_ms`
（計時，43/43 全不同——**這是計時本來就會動，不是結果漂移**）。

**「SFace control 不復現」這條 stop condition 未觸發。**

## 4. 同 SFace bytes bridge：成立

第一次量測**失敗**（cosine median −0.0156）。依 plan `:228`「先查
color／scale／resize／runtime」逐項查，**成因是 worker 的 scale 錯誤**：
`SFaceClient.forward` 內部為 `(img[0] * 255).astype(np.uint8)`，**它要
0–1 的 float BGR**，而餵入的是 uint8，等效 255×255。修正後：

```
n = 66（同一批 crop，產品 ORT 路徑 vs DeepFace cv2 路徑）
cosine_between_paths   min 0.99999994   median 1.0   max 1.00000006
max_abs_component_diff min 1.7e-07       median 2.4e-07  max 3.9e-07
66/66 在 1e-5 內
```

**同一份 SFace bytes 在兩條推理路徑下一致**，差異量級 2.4e-07 屬浮點雜訊。
**「bridge 不同 → 不可當模型提升」這條 stop condition 未觸發**；本報告的
兩臂差異不含 bridge 因素。

## 5. 兩臂結果

分母：probe 13、non-target 30。**errors 保分母** —— 兩臂各 43 列、
**unprocessable 0**（沒有被丟棄的列）。每格為 min／median／max。

**median 定義**：`statistics.median` —— n 為奇數取正中一列，n 為偶數取
中間兩列的平均。四捨五入至四位小數，下同。

| | SFace control | Facenet512 |
|---|---|---|
| **top1 正確 k／13** | **13/13** | **13/13** |
| probe wrong identity | 0 | 0 |
| probe top1 score | 0.4914／0.6636／0.7248 | 0.7291／0.8222／0.8789 |
| probe margin | 0.2853／0.4270／0.5232 | 0.1979／0.2756／0.3456 |
| non-target top1 | 0.3111／0.3972／0.5386 | 0.5477／0.6978／0.8625 |
| non-target margin | 0.0016／0.0338／0.1714 | 0.0037／0.0543／0.2523 |
| gallery | 23/23 | 23/23 |

> ⚠️ **兩列 `13/13` 必須連同下一段一起讀。**
>
> **probe 13 張全部是同一身分 `enroll-23`。** 因此「top1 正確 13/13」在兩臂
> 都是 13/13，**幾乎不含跨身分難度資訊**——它只證明「同一身分的 13 張
> probe 都能排到正確那一位」，不證明模型在 23 個身分間的判別力。
> **這是樣本限制，不是模型等價的證據。** 把這兩列當成「兩模型等價」，
> 正是 plan `:267`「post-hoc frontier 被當獨立準確率」那條風險的形狀。

**分佈才是差異所在。** Facenet512 的 probe 分數整體較高（median 0.82 vs
0.66），**但 non-target 同步更高**（median 0.70 vs 0.40，上限 0.86）。
non-target 有 **9/30** 的分數高於最低的 probe 分數，control 僅 **2/30**。

**這正是不可套用 SFace 門檻的實證**：把 0.363 套到 Facenet512，30 張
non-target 會全部落進 `matched`。

**本報告不宣稱 Facenet512 較好或較差。** 未選門檻、未做 frontier、未 sweep、
所有 `selected=False`。依 plan `:181`，可陳述離門檻多遠，
**但不得據此裁定應放寬多少**。分佈差異要讀成 accept／FA 邊界，
需要 accept/FA frontier 與**預先固定的獨立校準**，兩者本輪皆未做。

## 6. 限制

### 6.1 樣本（最重要）

- **probe 13 張全部是單一身分 `enroll-23`**（見 §5 的警示框）。
- **這不是本輪的取樣意外，是 corpus 結構。** 同一個 probe 身分在歷史 29 輪
  真機 session 裡 11 輪產生 top1，**全部**為 `enroll-23`（`PROJECT-STATE.md:39`、
  `:53` 記錄該 29 輪及其解讀界線）。**因此下一步不是換一批 probe，而是換 corpus
  或擴大 probe 身分覆蓋** —— 換 probe 身分不會改變任何一個 `13/13` 的意義。
- non-target 僅 30 張，corpus 僅 23 個註冊身分。
- **不外推 500 人**；不從 23 人 open-set 的結果推論產品規模下的表現。
- 單一 corpus、單一相機來源、單次採集，無跨 epoch／跨來源變異。

### 6.2 統計

- L 分支帶 g3-v1 品質 gate，**R 分支跳過 gate 但同樣要求單一可偵測臉**。
  兩者共用同一份 crop，**沒有把品質差異與模型差異混在一起**。
- 43 列的量級下，**任何比例數字的不確定區間都很寬**；本報告只給計數與
  分數區間，不給比率的統計顯著性。
- **沒有獨立校準集**。任何 accept／threshold 的數字都是 in-sample 描述。

### 6.3 ArcFace：已核准、artifact 已核、架構不相容（**不是**「環境不可行」）

- **已核准**（d-32）：限本機研究比較。
- **artifact 已核**：`6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93`
  （137,026,640 bytes）與 repo 記載逐位元組相符。
- **架構建構階段不相容**：`ArcFace.py:67 load_model → ResNet34()` 在
  **圖建構階段**拋 `AttributeError: 'KerasHistory' object has no attribute
  'layer'`，**早於載入權重**。
- **環境本身可行**：`Facenet.py:1687 load_facenet512d_model → InceptionResNetV1()`
  在同一環境可載入並完成 66 張量測。**是特定架構與該 Keras 組合不相容，
  不是 D6 環境無解。**
- **未降級套件試圖修復** —— 那要改依賴，屬 stop condition。**記錄不修。**
- plan `:86` 明載 ArcFace 為第二候選備選、**不預設必跑**，故**交付不受影響**。

### 6.4 環境

- **DeepFace 側 lock 不完整**：`uv.lock` 不含 `tf-keras==2.21.0` 與
  `retina-face==0.0.18`（import 期必要條件，事後 `uv pip install`）。
  **後果是雙向的：這些數字既不能被獨立重現推翻，也不能被獨立重現確認。**
  從 lock 重建會得到缺少這兩套件的環境，`import deepface` 直接失敗。
- 兩臂在不同 interpreter（3.14 / 3.13.15）。**禁止「在產品 Python 上跑」的
  表述**；本報告逐候選聲明。

### 6.5 已知的 harness 修正

Bridge 第一次失敗源於 worker 的 scale 錯誤（§4）。**該錯誤已修正並記錄**；
它不影響最終數字，但意味著 bridge 數字是修正**後**的量測。

## 7. 明確未做的事

- **未選定任何門檻**，未 sweep 產品 operating point，所有 `selected=False`。
- **未做 accept／FA frontier**（post-hoc 描述性 envelope）。
- **未套用 SFace 門檻**於候選。
- **未做獨立校準**，未以本批資料推導 margin／review 門檻。
- **未外推 500 人**，未宣稱模型等價。
- **未動產品 gate、three-frame 規則、gallery、learning、profile g3-v1、
  detector gate、依賴**（`pyproject.toml` / `uv.lock` 零 diff）。
- **未開相機、未連雲端人臉 API、未呼叫 `DeepFace.analyze`。**
- **未把逐張明細、crop、embedding、score 放入 repo。**

## 8. 證據位置與可重現性

**repo 外 run 目錄**（不進 Git）：
`~/Downloads/face_sample/_facecore/reports/d6-ac695befb042f7301320fa55ea5b1d8c15c9dc45/`

| 檔案 | 內容 |
|---|---|
| `corpus-manifest.json` | 三組張數 ＋ 聚合 digest（無逐張檔名） |
| `crops/crop-manifest.json` | crop 張數 ＋ 聚合 digest ＋ unprocessable 計數 |
| `d5-control.csv` | D5 control 43 列（19 結果欄 ＋ `stage_ms`） |
| `candidate-43-rows.csv` | 兩臂各 43 列（132 列） |
| `sface-bridge.json` | bridge 摘要 ＋ 逐張比對（repo 外） |

**可重現性（實測，非聲稱）**：重跑兩臂後，
`candidate-43-rows.csv` 的 **18 個欄位全部 bit-identical**。
該檔**不含計時欄位**。`stage_ms` 只存在於 `d5-control.csv`，
且 43/43 全數與前一次不同——**計時本來就會動，不參與任何比對**。

**唯讀 store 前後 hash 相同**（run 期間無任何寫入）：

```
bf0b53228364b394 mtime=1790666418 results.csv
c3b476165c5e535c mtime=1790744705 demo-results.csv
```

**授權邊界**：無二進位／權重／資料／照片進 Git
（`git diff --name-only` 對 `\.(onnx|h5|npy|csv|jpg|png)$` 零命中）。
