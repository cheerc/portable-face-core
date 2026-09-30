# D6 — DeepFace 候選授權與來源 Gate

- 檢索日期：**2026-09-30**
- Scope boundary（**2026-09-30 修訂，d-32 落地後**）：**只有唯讀 metadata 查證。** 截至本檔修訂時沒有下載
  Facenet512／ArcFace 權重、沒有安裝 DeepFace、沒有任何 DeepFace 真實推論結果。每個需要 artifact bytes
  才能成立的項目都標 `UNVERIFIED` 並寫明完成步驟，**絕不記為通過**。
- 範圍裁決：`d-20260930132146694094-31`；**權重裁決 `d-20260930132705762549-32`（operator 裁決 B）supersede
  其第 2 點** —— Facenet512／ArcFace 已核准**限本機研究比較**下載與使用。
- 來源計畫：[D6 計畫](../plans/2026-09-30-d6-deepface-comparison-plan.md) §5。
- 既有研究邊界：[open-source face stack](2026-09-09-open-source-face-stack.md) §「Not Safe as a Blanket
  Dependency Choice: DeepFace」—— DeepFace 可用於 non-shipping benchmark，**每個模型必須逐一批准**。
  operator 於 d-32 對 Facenet512／ArcFace 作了這個逐一批准，**同時明確接受其 license／provenance 未證實
  的風險**。
- **本檔記錄授權狀態，不是稽核流水帳。** 某候選取得 artifact 後，其 `UNVERIFIED` 欄位由實際下載前審查
  （downloaded-to-cache-then-hash）與 no-network-at-inference 證據更新；「未下載」本身不是待修的缺口。

## 與 `2026-09-10-model-candidate-gate.md` 的關係

兩份是**同類文件的兩個實例，並列不合併**：

| | 2026-09-10-model-candidate-gate.md | 本檔 |
|---|---|---|
| 對象 | 產品 bake-off（YuNet／SFace 選型） | D6 DeepFace 比較候選 |
| 候選集合 | YuNet 2023mar／2026may ＋ SFace fp32／int8bq | SFace control／bridge ＋ Facenet512／ArcFace |
| 授權邊界 | 產品出貨路徑 | **non-shipping 研究 benchmark** |
| 狀態 | SFace 已選入產品並落地 | SFace 沿用；兩個 DeepFace 新候選**已核准研究使用，license 未證實** |

時間軸與候選集合不同，屬並列的歷史記錄，**不是 source of truth 漂移**。本檔**不回頭更新舊檔**，
理由是：舊檔記錄的是 2026-09-11 對產品 bake-off 的授權判斷，追溯改寫會讓當時的決策依據不可重現。
D6 若最終採用新模型，應在本檔更新 D6 候選狀態並另開新的產品候選評估，而不是改寫 2026-09-10 的紀錄。

## Gate legend

- `CLEAR` —— 證據在檔且可經 locator 獨立複驗。
- `PROVENANCE_UNRESOLVED` —— 未被拒絕；保留在評估中並呈報 operator。
- `UNVERIFIED` —— 沒有 artifact bytes 或沒有本機 run 就無法成立；**從不計為通過**。
- `BLOCKED (未核准)` —— 被 operator 個別核准程序擋下；operator 裁決前不得下載／載入／執行。
- `APPROVED-RESEARCH-ONLY (未證實 license)` —— operator 已核准**限本機研究比較**使用，但該候選的 exact
  weight license／provenance **仍未證實**。**這兩件事同時為真：已核准使用 ＋ 授權未證實。** 不得把核准
  讀成 license 已解決。

---

## 候選 1 —— `ort-sface-control`（可用）

| 欄位 | 值 | 狀態 |
|---|---|---|
| 用途 | D5 control，固定門檻 0.363／0.3／0.10 | — |
| Model ID / weight locator | `face_recognition_sface_2021dec.onnx`，`opencv/opencv_zoo` `models/face_recognition_sface/` | `CLEAR` |
| 預期 SHA-256 來源 | `docs/mac-demo-baseline-d0.md:38` 凍結值 | `CLEAR` |
| Actual SHA-256 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | `CLEAR` |
| Bytes | 38,732（既有 artifact，本機 `~/facecore-models/`） | `CLEAR` |
| Code license | 同目錄 `LICENSE`，Apache-2.0 全文；README 稱 "All files in this directory are licensed under Apache 2.0 License." | `CLEAR` |
| Weight license | 同上目錄聲明涵蓋權重檔；**訓練資料權利另核**（見下 provenance） | `PROVENANCE_UNRESOLVED` |
| 允許用途／redistribution | Apache-2.0 §§1–9 grant use／reproduction／modification／distribution；**這是授權文字的記錄，不是法律意見**。redistribution 仍受 training provenance 條款 gate | `CLEAR`（文字層面） |
| Training provenance 已知 | SFace loss 訓練的 MobileFaceNet 實例；ONNX 轉換由 Chengrui Wang 自原始 code base；Zoo accuracy 0.9940(fp32) | `PROVENANCE_UNRESOLVED` |
| Training provenance 未解 | **opencv_zoo issue #313 自 2026-07-22 起 OPEN**：目錄 Apache-2.0 聲明是否延伸到 `2021dec` 權重參數的商業預測、哪個 corpus 產生該參數集、商用部署是否被來源資料規則允許。**無任何 maintainer 陳述把某 corpus 連結到這個檔案** | `PROVENANCE_UNRESOLVED` |
| Python／runtime／lock hash | 產品 Python `==3.14.*`、ONNX Runtime、root `uv.lock` | `CLEAR` |
| color／shape／normalization／dim | 112×112、128 維、L2-normalized；本 repo 既有 embedder 路徑 | `CLEAR` |
| generation | 既有 D0 凍結 generation | `CLEAR` |

## 候選 2 —— `deepface-sface-bridge`（可用）

| 欄位 | 值 | 狀態 |
|---|---|---|
| 用途 | **同模型、不同 wrapper 對照** —— 抓色彩／尺度／前處理接錯。不是模型比較 | — |
| Upstream pin | `serengil/deepface` `e1b38a8fced707b8acb137bcc0052302a3cae329` | `CLEAR` |
| Weight bytes | **同一份已核 SFace weight** —— 不引入新權重 | `CLEAR` |
| Weight SHA 對應 | 應等於候選 1 的 `0ba9fbfa…`；**實際比對待 E 產出**，不得假設 | `UNVERIFIED` |
| Wrapper code license | DeepFace repo license | `UNVERIFIED`（未取本輪 metadata） |
| 允許用途 | non-shipping 研究 benchmark | `CLEAR` |
| Training provenance | 同候選 1（同一份權重），含 #313 未解 | `PROVENANCE_UNRESOLVED` |
| Input contract 待查 | DeepFace ndarray API 是 **BGR**；SFace bridge normalization 暫定 `base` | `UNVERIFIED`（E 查證） |
| 推論網路 | `weight_utils.py` 只看 cache 存在、**無 checksum／offline 開關** —— 推論時必須先確保不會 auto-download | 待 E 證明 |

**同權重下若出現排序／數值差異，先查前處理接線**（BGR、resize 尺度、normalization、float→uint8），
不得直接歸因模型或 wrapper。

## 候選 3 —— `deepface-facenet512`（APPROVED-RESEARCH-ONLY：已核准研究使用，license 未證實）

| 欄位 | 值 | 狀態 |
|---|---|---|
| 用途 | 建議第一個不同 recognition 模型 | — |
| Weight locator | DeepFace release `v1.0/facenet512_weights.h5` | metadata 已知 |
| HF mirror | `huggingface.co/serengil/deepface`，revision `897aaade186abd2b9aeba45038a27d16c2df5f4e` | metadata 已知 |
| Metadata SHA-256 | `3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f` | metadata 已知，**未下載核算** |
| Code license | DeepFace wrapper | `UNVERIFIED` |
| **Weight license** | **exact license 未證實** —— wrapper 的 MIT **不補足**此缺口 | **`UNVERIFIED`** |
| **轉換來源鏈** | **未證實** —— H5 對應的訓練權重來源與轉換過程未確立 | **`UNVERIFIED`** |
| HF mirror 與 GitHub release bytes 是否相同 | **未證實** | `UNVERIFIED` |
| Training provenance | `UNVERIFIED` | `UNVERIFIED` |
| Python／runtime／lock | `UNVERIFIED` | `UNVERIFIED` |
| color／shape／normalization／dim | 160×160、512 維、`Facenet` normalization（per-image mean/std） | 待 E 查證 |
| 權重資格 | **operator 已核准**（`d-20260930132705762549-32` 裁決 B）：可下載與使用，**限本機研究比較** —— 不進 Git、不散布、不商用、不進產品 runtime。**核准不代表 license 已證實** | `APPROVED-RESEARCH-ONLY` |

## 候選 4 —— `deepface-arcface`（APPROVED-RESEARCH-ONLY：已核准研究使用，license 未證實）

| 欄位 | 值 | 狀態 |
|---|---|---|
| 用途 | 第二候選備選，**不預設必跑** | — |
| Weight locator | DeepFace release `v1.0/arcface_weights.h5` | metadata 已知 |
| HF mirror | 同上 revision `897aaade…` | metadata 已知 |
| Metadata SHA-256 | `6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93` | metadata 已知，**未下載核算** |
| Code license | DeepFace wrapper | `UNVERIFIED` |
| **Weight license** | **exact license 未證實**；**不得套用 InsightFace Buffalo_L 的授權** —— DeepFace 同名 ArcFace 不是同一個權重來源 | **`UNVERIFIED`** |
| **轉換來源鏈** | **未證實** | **`UNVERIFIED`** |
| HF mirror 與 GitHub release bytes 是否相同 | **未證實** | `UNVERIFIED` |
| Training provenance | `UNVERIFIED` | `UNVERIFIED` |
| color／shape／normalization／dim | 112×112、512 維、`ArcFace` normalization（`(x-127.5)/128`） | 待 E 查證 |
| 權重資格 | **operator 已核准**（`d-20260930132705762549-32` 裁決 B）：同候選 3 的範圍限制。**核准不代表 license 已證實** | `APPROVED-RESEARCH-ONLY` |

H5 與 PyTorch `.pth` 是不同 artifacts，**不能只記模型名**。兩者都要分別記錄 locator 與 SHA。

---

## 取得與核准政策

1. **不自行下載驗證後宣稱 license 已解。** metadata 缺口要明示，並在報告中保留「未證實」狀態。
2. 下載前按 root `CLAUDE.md:21` review 並記錄。**actual SHA 不符 registry 立即 fail，不得改 expected 值來湊。**
3. **下載前審查流程（d-32 第 3 點，強制）：downloaded-to-cache-then-hash。** 先下載到隔離 cache，
   再核 actual SHA，**核對通過後才允許推論**；推論階段 **no-network-at-inference**，不得 auto-download，
   empty cache 必須 fail-clear，不可靜默上網。
4. **repo 為 public（d-32 第 2 點）：只 commit 授權審查文件**（code license、weight license、provenance 記錄、
   下載 URL、checksum 策略），**不 commit 權重檔本身**，不寫入 license 或來源不明的第三方檔案。
5. 某候選 gate 未通過 **只阻擋該 candidate**，不阻擋 C（品質量測）與其他已核准候選的工作。
6. DeepFace 放在獨立實驗環境，與產品 Python 3.14／ORT 相依隔離。

## 未解事項

1. **Facenet512 H5 的 exact weight license 與轉換來源鏈** —— **仍未證實**。operator 於 d-32 接受此風險並
   核准研究使用；D6 報告必須明確記錄此項仍未證實。
2. **ArcFace H5 的 exact weight license 與轉換來源鏈** —— **仍未證實**，同上。
3. **HF mirror 與 GitHub release 的 bytes 是否相同** —— 未證實（影響 expected SHA 該記哪一個）。下載後以
   actual SHA 核對為準。
4. **SFace #313 未解**：目錄 Apache-2.0 聲明是否涵蓋 `2021dec` 權重參數的商業預測。此項不阻擋 D6 的
   non-shipping benchmark，但**任何出貨用途都仍被 gate**。
5. **gallery digest 無自動守護**：`static_baseline.py` 完全不碰 digest，`tests/` 裡唯一的 `e3d77c4c` 是無關
   fixture。D0 凍結值目前只是觀測值，D6 的 control 步驟必須人工核對。
6. **ONNX 替代候選（d-32 第 4 點，非阻塞）**：找 ONNX 授權明確、無 TensorFlow 依賴的替代模型候選
   （例如 MobileFaceNet ONNX、facenet-torch ONNX 匯出），列入**下一輪**候選 gate。**本輪不阻塞 A 系列**。

## 本檔狀態

截至 2026-09-30（本檔修訂時）：**沒有下載 Facenet512／ArcFace 權重、沒有安裝 DeepFace、沒有任何 DeepFace
真實結果。** 兩個候選已獲 operator 核准**限本機研究比較**使用（`d-20260930132705762549-32`），但其 exact
license／provenance **仍未證實** —— 上述未解事項 1–3 需在報告中保留此狀態，不因核准而消失。

下一輪（非本輪阻塞）：ONNX 授權明確、無 TensorFlow 依賴的替代候選。
