# D5 現版本靜態 baseline

用 App 現場實際在用的同一條 pipeline（同一個 gallery、同一個 detector gate、同一套品質篩選、
同一組 g3-v1 門檻）對 **13 張辨識組**與 **30 張 non-target** 逐張評分。執行於 2026-09-30。
計畫：[D5 現版本靜態 baseline 執行計畫](../plans/2026-09-30-d5-static-baseline-plan.md)。

## 1. 版本資訊

| 項目 | 值 |
| --- | --- |
| 程式 HEAD | `cf1d6dd0b3f4238ce333f961b74efa24c9e5b820` |
| interpreter | CPython **3.14**（`requires-python = "==3.14.*"`，`pyproject.toml`）——產品自己的直譯器，control 與 App 同環境 |
| onnxruntime | **1.30.0**（pin 自 D5 產出 commit 起未變，見下） |
| execution provider | **`CPUExecutionProvider`，原因是產品 embedder 硬寫**，不是觀測值（見下） |
| `ALIGN_CONTRACT_VERSION` | 3（`pipeline/align.py:25`） |
| detector | YuNet 2023mar，gate = `PolicyProfile.frozen_v1()` 的 `detector_confidence_min` 0.90 |
| profile | `g3-v1`：match 0.363／review 0.3／margin 0.10 |
| gallery digest | `e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f` |
| gallery | 23 個身分，generation `gen-1` |
| model generation（embedder） | `face_recognition_sface_2021dec` |
| YuNet sha256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| SFace sha256 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

**gallery digest 與 D0 凍結值逐字相符**，代表本報告的評分對象與 App 現場看到的是同一個
註冊組。這是 plan 的 H1，本報告成立的先決條件。

### 為什麼 ORT 版本是可查的事實，而不是 Unknown

D5 產出時並未記錄當時的 ORT 版本，這是產出端的遺漏，不是當時無法得知。`onnxruntime==1.30.0`
這個 pin 在 D5 **harness commit**（`cf1d6dd0`，PR #126）與**報告 commit**（`5924939`，PR #127）
的 `pyproject.toml:16` 都是同一個值，且 `git log cf1d6dd0..main -- pyproject.toml` 為空。
（`cf1d6dd0` 是 `5924939` 的祖先，不是相反——報告 commit 較晚。兩個 commit 都要查，是因為
本報告 §1 自己記錄的「程式 HEAD」是 harness commit `cf1d6dd0`，而「產出這份報告」發生在
`5924939`；只寫後者會讓讀者照字面追一個比報告本身更晚的祖先。）
因此「該次 run 用的是 1.30.0」是 repo 事實，不需要用今天的環境去回推。

### 為什麼 execution provider 寫「硬寫」而不是「有 CoreML 可用」

venv 裡 **三個 EP 都可用**：實測 `ort.get_available_providers()` 回
`['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`。
但「venv 有 CoreML」與「這次 run 用了 CoreML」是兩件事。實際選擇由程式碼決定：
D5 走 `static_baseline.py` → `_build_true_context`（`research/cli.py:623`）→ 產品 `Embedder`
→ **`pipeline/embed.py:42` 寫死 `providers=["CPUExecutionProvider"]`**；detector 同理，
`pipeline/yunet.py:159` 也是寫死 CPU。

**所以這一欄是硬寫碼決定的結果，不是執行期觀測。** 記成「venv 有 CoreML 所以可能用 CoreML」
會讓未來讀者以為 D5 的數字來自 CoreML 路徑，事實上它來自 CPU 路徑——而 M 的
ORT-Mobile 結果（[`2026-09-30-d6-m-onnx-mobile-usability.md`](2026-09-30-d6-m-onnx-mobile-usability.md)）
證明同一份權重在 NNAPI 與 CoreML NeuralNetwork 上是 100% 覆蓋的，也就是說
**D5 的數字不能被外推成「在 NNAPI／CoreML 上也一樣」**：那是另一條 EP 路徑，尚未在 23／13／30 上量過。

## 2. 辨識組（truth = enroll-23，13 張）

| 類別 | L 分支（App 看到） | R 分支（跳過品質篩選） |
| --- | --- | --- |
| 正確接受 | **12/13** | **13/13** |
| 認錯人 | 0/13 | 0/13 |
| review | 0/13 | 0/13 |
| 未知拒絕 | 0/13 | 0/13 |
| 品質拒絕 | 1/13 | 0/13 |
| 無法處理 | 0/13 | 0/13 |
| top1 排序正確（不論 band） | 12/13 | 13/13 |

**L 分支與 R 分支的差異全部來自那 1 張品質拒絕**：L 把它算品質拒絕，R 仍排出
`enroll-23` 並 matched。**兩個分支都沒有認錯人。**

## 3. non-target（30 張）

| 類別 | L 分支（App 看到） | R 分支（跳過品質篩選） |
| --- | --- | --- |
| **FA（現行門檻 0.363／0.10）** | **2/30** | **2/30** |
| review | 6/30 | 28/30 |
| 正確拒絕 | 0/30 | 0/30 |
| 品質拒絕 | 22/30 | 0/30 |
| 無法處理 | 0/30 | 0/30 |

### 8 張品質通過的 non-target 各自命中不同成員

43 列中有 20 列通過品質篩選並帶有 top1：其中 12 列（辨識組品質通過的那些）
命中 `enroll-23`，另外 **8 列各自命中 8 個不同的 gallery
成員** —— 全部是 non-target 且品質通過，其中 6 列落在 review
（分數過門檻但 margin 不足）、2 列落在 matched
（即 FA 2/30）。

**這說明 FA 不是隨機誤判，而是「非目標被匹配到某個相近的註冊組成員」。** 本報告不逐張點名
身分（那是逐張明細，留在 repo 外的 CSV）。

### 「正確拒絕 0/30」的含義

**「正確拒絕 0/30」搭配「品質拒絕 22/30」是本報告最容易被誤讀的一格。** L 分支下
**沒有任何一張 non-target 是被模型判為 unknown 的**：30 張要嘛被品質篩選擋掉（22）、
進 review（6）、或被接受（2）。**模型的排序能力在這個門檻下幾乎沒有機會表現** ——
這是 D0 §11 第 18 項陷阱的另一個面向（那一項是「標註為 unenrolled 卻分數很高」，
這一項是「品質篩選先擋掉大多數，剩下的又都落在 review」）。**本報告陳述分佈，不推論成因。**

## 4. 分數區間

不寫逐張分數；以下為 min／median／max。

| 集合 | 分支 | 欄位 | min | median | max |
| --- | --- | --- | --- | --- | --- |
| 辨識組 | L | `l_top1_score` | 0.4914 | 0.6596 | 0.7248 |
| 辨識組 | L | `l_margin` | 0.2853 | 0.4216 | 0.5232 |
| non-target | L | `l_top1_score` | 0.3787 | 0.4000 | 0.5386 |
| non-target | L | `l_margin` | 0.0251 | 0.0398 | 0.1714 |
| 辨識組 | R | `r_top1_score` | 0.4914 | 0.6636 | 0.7248 |
| 辨識組 | R | `r_margin` | 0.2853 | 0.4270 | 0.5232 |
| non-target | R | `r_top1_score` | 0.3111 | 0.3972 | 0.5386 |
| non-target | R | `r_margin` | 0.0016 | 0.0302 | 0.1714 |

non-target 的 L `margin` 下界 0.0251 遠低於門檻 0.10，
R 分支下界更低到 0.0016 —— 這正是 30 張裡只有
2 張能達到 matched 的原因。

## 5. 品質拒絕原因的分布

依 reason code 計數，共 23 張品質拒絕：

| reason code | 張數 |
| --- | --- |
| `quality_exposure` | 23 |

**這個 reason code 同時涵蓋兩個條件**（`pipeline/quality.py:42-46`）：平均亮度超出 40–215
區間（太暗或太亮）**或**過曝像素比例 > 0.05。**本報告只寫「23/23 的品質拒絕是曝光相關」這個事實**，
不推論照片本身是否偏暗 —— 單一批次、單一場景、沒有光照對照，**照片偏暗與量測誤判無法從
這批資料分辨**。

這落在 D0 §11 第 3 項（曝光或品質拒絕是場景問題還是量測／門檻問題）正中央。該項現況是
「部分已驗，樣本不足以斷言」。**D5 讓那一項更清楚 —— 但沒有回答它。**

## 6. 延遲

| 階段 | n | min (ms) | median (ms) | max (ms) |
| --- | --- | --- | --- | --- |
| detection | 43 | 12.68 | 21.58 | 30.78 |
| quality | 43 | 3.57 | 11.73 | 29.23 |
| embed | 20 | 6.29 | 7.02 | 13.46 |

⚠️ **靜態照片的解析度與相機畫面不同，這組數字不代表現場延遲。** 單張照片無追蹤與連續取樣
成本，且 embed 只在品質通過時發生（本次 n=20，
其餘 23 張品質拒絕在 embed 之前就結束）。

## 7. sweep grid（live 語意，只作呈現）

grid 為既有 `REAL_MATCH_GRID`（0.30–0.60）× `REAL_MARGIN_GRID`（0.05–0.20），以 import 取值。
共 28 格，每格以 live 的 band 規則重算兩個分支。本報告**不從中挑任何一格作為新門檻**：
D5 沒有 holdout，這批照片**就是**評估集，事後從 grid 挑門檻等於用測試集調參。
實測 `selected` 欄位全部為 False（`any(selected) = False`）。

**22/30 的品質拒絕不會因任何門檻組合而改變。** 機制上要注意措辭的精確性：那 22 列在
`sweep_live_semantics` 裡是因為 `l_top1_score is None` 而 **`continue` 跳過**（`static_baseline.py:574`），
**它們從未進入 sweep 的計數**，而不是被 sweep 判定為「與門檻無關」。兩者結論相同
（這 22 列在任何門檻組合下都不會出現在 FA 欄），但機制不同：若日後有人把跳過改成
「計為 unknown」，本段這句話就需要更新。因此 sweep 的 FA 欄只會在剩下的 8 張
品質通過的 non-target 上變動。

## 8. 限制

1. **資料小且曾被檢視。** 13 張辨識組全部來自同一人；30 張 non-target 是單一批次。
   本報告的任何比率都**不得外推**到其他人或 500 人規模，也**不得當作獨立的泛化證據**。
2. **靜態結果不推論動態。** 三幀規則不適用於單張照片；本報告**不**說明動態 session 的表現。
   D4 的動態觀察（32 輪）是另一組資料，見 D0 §11 第 17 項。
3. **歷史數字協議不同，不得並列比較。** M1（FA 4/30）是 `ALIGN_CONTRACT_VERSION=1`、
   detector gate 0.8 下量的；M5 是 margin 0.15／match 0.30–0.55 的二維表；v3 重跑
   （M1 1/30）的 gallery digest 與 live 不同。**三者的 gate、contract、gallery digest 都與本
   報告不同，只能當脈絡，不得讀成「變好了」或「變差了」。**
4. **R 分支比 App 樂觀。** R 跳過品質篩選，**App 不會這樣判定**。R 分支只用於把「模型認不出」
   與「品質篩選擋掉」分開看，不可當成產品行為。
5. **門檻未被選定。** 本報告呈現 sweep 但不挑門檻；調門檻需要 holdout 或新資料集。
