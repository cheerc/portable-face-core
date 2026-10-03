# D7-A：G3 診斷性 log 與啟動健壯性（計畫修訂草案 v8）

- 日期：2026-10-01
- 作者：fc-team-lead（依 fc-team-commander v1 ＋ team-discuss 三份獨立探索收斂）
- 狀態：**已封盤（v8）。實作派工已授權並執行中** —— W3（#138 `51bf483`）／W0-a（#139 `f369e0a`）／W1（#142 `2059380`）已合併；W3 follow-up 註解修正（#144）**仍 OPEN，卡在 #141 operator 裁決**。剩餘 **W2 → W5 → W4**（W4 必須最後做）。
  ⚠️ **本行狀態由 Lead 於 2026-10-03 更新**（檔案納入 git 版控時，見 commit `28fb299`）。⚠️ **其餘 580 行自封盤後未變更** —— 下文的規格、順序與驗收條件仍是 W2／W5／W4 的權威依據。
- 建議 repo 落點：`docs/plans/2026-10-01-d7-g3-diagnostic-log-plan.md`
- ⚠️ **本檔是拋棄式執行文件**：功能權威是 spec 樹；執行完不維護、不回填。本檔不構成產品 API 承諾。

---

## 0. 本次修訂的核心：v1 的結論方向要翻轉

v1 §0 的定位是「**現在沒有可調的參數空間**，operator 的訴求需要先被診斷才能被滿足」。

**discuss 的三份獨立探索一致顯示這個結論的因果歸屬是錯的，且會讓 operator 放棄。**

**修訂後的定位：門檻在這批 log 上從未成為否決者，但那是因為那批 log 裡一個冒名者都沒有。真實的可分性差距現在就量得到（產品條件下 genuine 13/13 vs impostor 2/30），而 operator 的原始訴求有可執行入口——只是 v1 指錯了參數。**

---

## 1. 實測基礎（三方獨立重算 + lead 第四方抽驗）

### 1.1 log 基本結構

`~/Downloads/face_sample/_facecore/store/demo-results.csv`

| 項目 | v1 敘述 | 三方獨立重算 | 判定 |
|---|---|---|---|
| 列數 | 32 | 32 | ✅ |
| **欄數** | **24** | **23**（source `G3_DEMO_RESULTS_CSV_COLUMNS` 亦 23） | ❌ **v1 錯，須改** |
| `result` | invalid_input 16／timeout 5／matched 11 | 同 | ✅ |
| `label_kind` | unlabeled 12／uncertain 7／unenrolled 5／enrolled 8 | 同 | ✅ |
| `frames_sampled==0` | 11／32 = 34% | 11／32 = 34.4% | ✅ |
| **`elapsed_ms` 中位數** | **15205.3** | **15182.2**（15205.3 是資料裡某一列零幀輪的值，非中位數） | ❌ **v1 錯，須改** |
| matched `top1_score` | 0.6219–0.7279 | 同 | ✅ |
| matched `margin` | 0.3692–0.5080 | 同 | ✅ |

### 1.2 決定性發現：這 32 輪只有一個身分

```
top1_identity : {'': 16, 'enroll-23': 16}
label_identity: {'': 24, 'enroll-23':  8}
有分數的 16 列 → top1_identity 全部是 'enroll-23'
```

**0.6219～0.7279 量的是同一個身分重複 11 輪的組間變異（**sd 0.0380 樣本／0.0362 母體**），不是身人間的距離。**

⚠️ **這是 repo 裡已有的教訓被重新發現**：`docs/PROJECT-STATE.md:130` 記載「歷史 29 輪……11 輪有 top1 亦全為 `enroll-23`。**不支撐跨身分判別力**」；`:180` 另寫「probe 13 張全部是單一身分 `enroll-23`」。**v1 §1.4 差點重新引入 `PROJECT-STATE.md` 已經明確否證的錯誤。**

### 1.3 真正無 usable frame 的輪次是 50%，不是 34%

```
frames_usable == 0 : 16 / 32 = 50.0%

reason_codes 分布：
  11  no_frames_captured|timeout                          ← 相機層失敗
   2  all_frames_rejected_quality|deadline_exceeded       ← 品質門過濾
   2  all_frames_rejected_mixed_causes|deadline_exceeded  ← 品質門過濾
   1  all_frames_rejected_no_face|deadline_exceeded       ← 品質門過濾
```

⚠️ **v1 把 34% 稱為「34% 的輪次完全沒有畫面輸入……是相機層面的失敗，不可能靠任何模型或門檻調整改善」——這個表述把 15.6% 的品質門過濾誤歸為不可改善。** ⚠️ **但「品質門過濾可改善」同樣錯——品質門 code-frozen、無旋鈑可調（見下方 L72 的分類）。**

**修訂**：v1 §1.2 的標題與結論須改為「34% 無畫面輸入（相機層）＋ 16% 品質門全拒＝ 50% 無 usable frame」。

⚠️ **⚠️ 那 16% 的「可改善」前提不成立，須改為「無旋鈑」。** 實測 `profiles/g3-v1.json` **沒有任何品質門欄位**（只有 `quality_policy_version`）；`cli.py:734-744` 的 `profile_to_policy` 只傳 `match_threshold`／`review_threshold`／`margin_threshold` 三個門檻，**七個品質門（`quality.py:32-51` 的 `quality_detector_confidence`／`quality_face_too_small`／`quality_blurry`／`quality_exposure`／`quality_pose_yaw`／`quality_pose_pitch`／`quality_occluded`）code-frozen 在 `PolicyProfile.frozen_v1()`**。

**operator 無法透過 `profiles/` 調整它們**——要「改善」得改 `frozen_v1()`，而那是 §6 技術邊界明文禁止的。

⚠️ **這是 §4 不可轉家族的無名成員**：七個品質門接受設定、算進 `quality_policy_version`、卻無 plumbing。⚠️ **它與 `max_frames` 不同族——處置完全相反**：品質門要改 `frozen_v1()`（§6 禁止）；**`max_frames` 是 by design、有測試釘住、修不了也不該修。**

**所以正確的分類是：34% 不可改善（無旋鈑，相機層）＋ 16% 不可改善（無旋鈑，品質門 code-frozen）＝ 50% 全部無旋鈑。** v1 把 16% 誤歸為「可改善」是錯的——**但 v1 把它併入「不可改善」也是對的**，因為它同樣不可改善。**v2 的「可改善」判斷才是錯的。**

**零幀 11 列的耗時是雙峰的**（impl 實測）：9 列在 6–18 s、2 列在 71–73 s。**「34% 零幀」是異質的，不該當成單一故障率讀。**

### 1.4 ⭐ 門檻：現象成立，因果歸屬錯誤

**成立的部分**（三方一致 + lead 抽驗）：

門檻確實從未成為任何一列的否決者。lead 抽驗 `result` 對門檻的敏感性：**門檻升到 0.5 或 0.6，這 32 列的 `result` 一列都不變**（唯一在 0.6 下被排除的 `s=0.5621` 那列本來就 `timeout`，被 support 擋住，門檻改變只會讓它「更早被 reset」）。

**但因果歸屬是錯的。** impl 用磁碟上的真實 corpus 跑了冒名者／本人配對（真 `YuNetDetector` ＋ `SFace`，非 mock）：

```
IMPOSTOR non-target : n=30  best-sim min=0.3111  p50=0.3972  max=0.5386
GENUINE enroll-23   : n=13  best-sim min=0.4914  p50=0.6636  max=0.7248

套用門檻：
| thr | **只套 score 門**（非產品路徑） | **score ＋ margin 門（產品語意）** |
|---|---|---|
| 0.363 | genuine 13/13 ｜ impostor **24/30（80.0%）** | genuine 13/13 ｜ impostor **2/30（6.7%）** |
| 0.500 | genuine 12/13 ｜ impostor 2/30（6.7%） | genuine 12/13 ｜ impostor 1/30（3.3%） |
| 0.600 | genuine 11/13 ｜ impostor 0/30（0.0%） | genuine 11/13 ｜ impostor 0/30（0.0%） |

⚠️ **⚠️ 兩欄條件不同，引用時必須標明是哪一欄。** 產品路徑同時套 `margin_threshold`（`cli.py:734-744` `profile_to_policy` 傳入三個門檻），**而原 sweep 只套 score 門**——`24/30` 那個數字來自去掉 margin 門的量測。

**30 張 non-target 中只有 2 張 margin ≥ 0.10**：
```
nontarget-16  top1=0.5386  margin=0.1714  PASS
nontarget-07  top1=0.4122  margin=0.1062  PASS
nontarget-29  top1=0.5274  margin=0.0683  fail   ← 分數高但 margin 不足
nontarget-08  top1=0.4687  margin=0.0466  fail
```

⚠️ **所以「0.363 會讓 24/30 冒名者通過」這句是錯的**（那是 score-only 條件）。**採產品語意時 0.363 下只有 2/30 會通過。**

**但 §0「門檻有空間」的結論依然成立**：產品語意下 2/30 = 6.7%，調到 0.55–0.6 即歸零，而真分損失僅 1–2/13。

⚠️ **這正是 §4 立下要防的形狀：用比產品條件寬鬆的度量支撐關於產品的結論。**

⚠️ **30 張 non-target 有 24 張（80%，score-only）落在 0.363 以上；產品語意（＋margin 門）下只有 2/30。門檻不是「不作用」，是這 32 輪永遠碰不到它的上界——因為沒有冒名者。**

**真正的可分性不足現在就看得到**（genuine min 0.4914 與 impostor max 0.5386 重疊），但它藏在 operator 沒測過的那 30 張裡，**不需要外推到 500 人規模**。

**修訂後的 §1.4**：

> 門檻 `match_threshold=0.363` 在本 log 的 32 輪**單一身分**資料上從未成為否決者；這不能推論為「門檻沒有可操作空間」。**「門檻不作用」是樣本結構的性質，不是參數的性質。**

⚠️ **⚠️ 引用上表時必須標明欄位。** score-only 欄（24/30）是**去掉 margin 門**的量測，**不能用於任何關於產品的結論**。

⚠️ **margin 門的套用位置**：`session.py:435-438` 直接讀 `self.profile.match_threshold` 與 `self.profile.margin_threshold`。**`SessionEngine` 持有整個 `ResearchProfile` 物件**（`session.py:56` `self.profile = profile`，`cli.py:1108` 原樣傳入，無逐欄位解包）。**不是 `profile_to_policy`** —— 後者只產出給 `ScoringContext`（research 模式，`cli.py:694`）。

**這使 §2「不選產品門檻」的非目標成立，但理由不是「FAR 很高」**：

產品條件下，門檻 0.363 → 0.5 只把冒名者從 **2/30 降到 1/30**（邊際效益不足一張），而 0.5 → 0.6 才歸零、真分損失 2/13。**沒有任何單一門檻調整能同時壓住 FAR 與保住真分。** 真正的判斷依據是 support／品質門的聯合條件——⚠️ **單張 FAR 已量得（見上表產品欄），但 support 聯合條件量不出來，見 §7 premise 6。**


⚠️ **⚠️ 「成本極低可量 support 聯合模型」是錯的，已刪。** 實測：corpus 中**每人只有 1 張照片**，而 `required_support=3` 需要**同一輪 3 個不同時間點的影格**——**靜態語料推不出 support 效果，只有真機連續擷取能量得到。** 便宜的是**已量過的單張 FAR**；需要聯合模型的那件事從這個 corpus 量不出來。

**交叉驗證**：impl 獨立重現的 SFace non-target top1（0.3111／0.3972／0.5386）與 D6 報告（`docs/research/2026-09-30-d6-deepface-comparison.md`）**完全一致**——兩條獨立路徑產出同一組數字。

### 1.5 真正 binding 的是三幀規則，不是門檻

5 列分數 0.5621～0.6995（**遠高於 0.363**）且 margin 0.3101～0.4279（遠高於 0.10），**全部通過兩個分數門檻**，卻仍失敗：

```
s=0.6677 m=0.4279 usable=1  insufficient_evidence|support_0_of_3|best_baseline_matched
s=0.5621 m=0.3101 usable=1  insufficient_evidence|support_0_of_3|best_baseline_matched
s=0.6649 m=0.4136 usable=2  insufficient_evidence|support_0_of_3|best_baseline_matched
s=0.6353 m=0.3811 usable=3  insufficient_evidence|support_0_of_3|best_baseline_matched
s=0.6995 m=0.4230 usable=6  insufficient_evidence|support_1_of_3|best_baseline_matched
```

⚠️ **`best_baseline_matched` 是關鍵 reason code——單幀已達 matched 帶，被 multi-frame 規則擋下。**

**operator 要調的是 `required_support` 與 `min_support_interval_ms`，不是 `match_threshold`。** r18 有 3 張 usable frame 卻 `support_0_of_3`，正是間隔條件（`session.py:483` 的 `interval_skip`）。

### 1.6 4 列標記矛盾：**兩個並存假設，不是已診斷

`label_kind=unenrolled` 共 **5** 列（v1 寫「4 列 `label_kind == unenrolled`」，混淆了兩個數），其中有分數的恰為 4 列，第 5 列是零幀列、本來就無分數。

⚠️ **v1 §1.6 的因果斷言「這不是程式 bug，是 operator 在畫面中為 `enroll-23` 卻按了『未註冊』」是未經驗證的斷言，且較不可能。** 另一個同樣符合資料的解釋是：**operator 真是陌生人、系統發生誤認**——而 §1.4 已量到 non-target 在 0.363 下 24/30 會通過，**誤認在這個門檻下是預期行為，不是例外。**

**⚠️ 修訂：不是兩個並存假設，是三個——而且「誤標」這個假設本身缺乏證據基礎。**

**實測：`label_kind` 是終局狀態的確定性函數，不是 operator 對「畫面中是誰」的獨立判斷。**

```python
# qt_window.py:1498-1506
if correct:
    identity = self.desktop.display_identity()
    if identity is None:
        label_kind = "unenrolled"     ← 按「正確」但系統沒顯示名字
    else:
        label_kind = "enrolled"

# desktop.py:331-337  display_identity()
if self._terminal.status == SessionStatus.matched:
    return self._terminal.matched_identity
return None                            ← 非 matched 一律 None
```

**全部 32 列驗證這條規則，0 例外**：`unenrolled` 列的 `result` 全是 `invalid_input`／`timeout`，`enrolled` 列全是 `matched`。

⚠️ **所以那 4 列的 `unenrolled` 記錄的是「系統說找不到，operator 按了正確」——operator 從未被問「這輪是誰」。** `label_identity` 全為空字串正與此一致。

**三種並存假設：**
1. **系統的 not-found 判定本身可疑**（最可能）——同一批 support 失敗的輪次，UI 顯示「找不到」，operator 如實標記
2. **真誤認**——operator 真是陌生人，系統發生誤認
3. **誤標**——operator 誤操作。⚠️ **此假設缺乏證據基礎，因為 operator 根本沒做那個判斷。**

**W4 的摘要工具必須輸出三種解釋各自的 k/N**，不得替 agent 收斂到一種。

⚠️ **W0 的理由應收斂（v5 的版本過強）：**

實測交叉表顯示 **對系統匹配成功的 8 列，`label_identity` 已經是呈現者身分**——**跨身分判斷在 matched 輪次上已經做得到**。

**缺口在未匹配輪次**（4 列 `timeout` ＋ `unenrolled`，`label_identity=''`）。⚠️ **而那正是 FAR 的證據所在** —— 決定門檻該不該動的是那些沒匹配上的輪次，**它們的呈現者是誰，log 裡一個字都沒有。**

**所以 W0 的理由是：「matched 輪次已有 ground truth，未匹配輪次沒有——而後者才是 FAR 的證據所在。」** 這個理由比「log 無法支撐任何跨身分判斷」更窄，但**更精確、也更有說服力**。

### 1.7 gallery 完整性：**已由第一手證據滿足，不掛在 W1**

impl 與 reviewer 各自用真 `YuNetDetector` ＋ 真 `SFace` 走 demo 同一路徑重建：

```
expected_count: 23  loaded_count: 23  failures: []
gallery_digest: e3d77c4cfab5b141a8abaecdb908254d7558f747a97395acd138072ded72141f
```

**與 32 列 log 裡的 `gallery_digest` 逐位元組相同。** 註冊組 mtime 早於 log，重建即當時狀態。

**結論：那 32 輪的 gallery 確實是 23/23 完整、無人被排除。§1.4 的 conditional 前提已滿足。** W1 的價值在**防未來**，不是補這批資料的缺口。

### 1.8 digest 對組成的敏感性（**更正 v1 與中途補充**）

`frame_pipeline.py:267` 只雜湊 `sorted(embeddings.keys())` 與其 embedding bytes。

- ✅ **被拒照片不在輸入裡，所以 digest 不給「原因」。**
- ✅ **但組成變動必然改變摘要**（lead 實測：23 keys → `016679609d249097`；22 keys → `a6536fc032d90cbb`）。

**正確描述是「digest 回答『載入集合變了沒有』（是／否），但不回答『少幾個、為什麼』」**，不是「對完整性沉默」。

⚠️ **但它有一個不可由 digest 補上的缺口**：digest **相同**時無法區分「一直是完整的 23 人」與「一直是缺人的 22 人」——因為缺人的那一次產生的 digest 本身就是那個 22 人的 digest。**所以 log 永遠無法重建當時的 `loaded_count`**，這正是 W1 那三欄存在的理由。

⚠️ **本輪的 gallery 重建證明什麼、不證明什麼**：實測重建得到 `e3d77c4c…` 與 log 相同，這證明「當時的 loaded 集合雜湊值 ＝ 現在的」，**但它依賴註冊組資料夾自 2026-09-11 未變動**（mtime 早於 log 的 09-30）。**若 operator 在那之間刪過一張又補回一張，digest 也會相同。** 加上 `load_report` 的 `loaded 23/23 failures=()` 才組成完整證據——**而 `load_report` 只在 live rebuild 時可得，log 永遠給不了。**

### 1.9 啟動路徑非持久註冊，但檢查完整

`cli.py:678` 走 `build_gallery_from_folder(..., generation=TRUE_PIPELINE_GENERATION)`，**每次啟動從 23 張重算，不寫 store**。

**完整性檢查存在且完整**（`frame_pipeline.py:166`）：逐張 解碼 → `detect` → `enforce_single_face` → `align_crop` → `embedder.embed`；單張失敗收進 `failures`（`EnrollmentFailure(filename, reason)`，帶檔名與原因）；重複身分 raise；全數失敗才整體 raise。另有 `strict` 模式。

**但 `strict` 預設 `False`，唯一呼叫端 `cli.py:679` 未傳 `strict`** → **部分照片被拒時 App 照常啟動，只是 gallery 少人。**

⚠️ **可見性存在於 UI，但與 log 脫鉤**：`qt_window.py:526-537` 的 `_update_enrollment_ui` 已顯示「註冊組：應載入 N 人，實際載入 M 人」＋逐張失敗原因；**但 `:326` 是 `load_report or getattr(gallery, "load_report", None)`，沒有綁到辨識輪次**。demo CSV 只有 `gallery_digest`，**無 `expected_count`／`loaded_count`／`failures` 欄。**

**啟動成本實測**：gallery 重建 23 人 = **1.34 s**（YuNet cold 112 ms ＋ SFace cold 54 ms）。**對 operator 目前的迭代節奏不構成瓶頸。**

---

## 2. ⭐ W0（新增，最優先）：診斷 run 必須有實驗設計

**v1 最大的缺口不在 W1–W5，在它們的前提：那 32 輪是 free-run，沒有實驗設計。**

**沒有 W0，D7-A 交付的是一套更精確地測量單一身分的診斷基礎設施，而 operator 的核心問題（門檻該不該調、調多少、模型換不換）依舊答不出來。**

**W0 內容**：診斷 run 的組成必須明文規定——

1. **至少 N 個不同註冊身分輪替入鏡**（不是同一人跑 16 次）
2. **M 張 non-target 輪替入鏡**
3. **每一輪明碼標示「這輪呈現的是誰」**（operator 按了什麼、以及系統判成什麼）

**成本極低——是 runbook 與操作流程，不是程式碼。**

⚠️ **⚠️ W0 必須拆兩半，因為「最優先」與「不在授權內」是真矛盾（review 兩份一致指出）：**

| | 內容 | 授權 |
|---|---|---|
| **W0-a** | W0 實驗設計寫成 runbook／文件（跨身分輪替、non-target 輪替、明碼標示呈現者、每輪的 `probe_kind`／`presenting_identity` 記錄規則）。**不碰機器。** | ✅ **本輪授權內** |
| **W0-b** | operator 在場執行該 runbook，產出跨身分 log。**需要真機操作。** | ❌ **需另行 go**（§6 排除真機量測） |

⚠️ **W0 不在授權內的真因不是「它是 runbook 不是程式碼」（runbook 不需授權即可寫），而是「執行 W0 需要 operator 在場操作真機」。**

⚠️ **時序上「最優先」不成立**：W0-b 產生的資料必須靠 W3 的 `probe_kind`／`presenting_identity` 才可分析，**所以 W0-b 必須等 W3 落地**。正確順序為 **`W3 → W0-a → W0-b`**。

⚠️ **W0 必須與 W1–W3 的 log 欄位一起設計**：若 log 不記錄「本輪呈現者」與「是否 non-target」，W0 產生的資料仍然不可分析。

**W0 需要的 log 欄位（最小集）**：`probe_kind`（`target`／`nontarget`）、`presenting_identity`（operator 主觀標記）。

---

## 3. 工作項（修訂）

### 3.1 W1：把既有 gallery 可見性綁到 log（**不是寫新檢查**）

**性質**：接線，非新功能。`GalleryLoadReport`（`expected_count`／`loaded_count`／`failures`）已存在且已序列化到 UI。

- **W1-a**：把 `expected_count`／`loaded_count`／`gallery_rejected` 寫入 demo log 的每一輪。
  ⚠️ **第三欄建議存 `filename（reason）` 形式、空字串代表全成功**（`EnrollmentFailure` 已序列化好）——**只存數量不存「是哪些」，就重現 §1.6 原本的問題**。
- **W1-b**：`strict` 是否在 demo 模式開啟——**這是產品決策不是實作，需另行裁決**。⚠️ 建議**不開**：`strict=True` 會讓一張壞照片就拒絕整個 gallery、App 不開，對實驗迭代更差。

**不應升級為阻塞項**：§1.7 已證那批 log 未被污染；W1 的價值在防未來。

### 3.2 W2：只補真正缺的——11 列零幀的成因分類

**v1 的 W2 範圍過大。** `all_frames_rejected_*` 那 5 列**已經有分類**（reason code 已區分 quality／mixed／no_face）。

- **只需補**：`no_frames_captured` 那 11 列的成因（相機開啟失敗／permission／被佔用／啟動時序）。
- **成因位置已定位**：`timeout` token 由 `controller.py:369` 與 `controller.py:851` 兩處發出，**不區分「相機從未開啟」與「相機開了但沒交出影格」**。
- ⚠️ **注意這 11 列的耗時是雙峰的（9 列 6–18 s／2 列 71–73 s），分類需能反映這個差異。**

### 3.3 W3：搬出既有資料，不是設計新欄位

⚠️ **v1 §3.3 是最大的過度設計：它在重新發明 repo 裡已經存在的東西。**

**`SessionResult` 已經帶、已經填、只是沒寫進 CSV 的欄位：**

| 既有欄位 | 位置 | 對應 v1 想解決的問題 |
|---|---|---|
| `timing_marks` 的 `recognition_duration_ms` | `contracts.py:365`，`to_dict()` 已序列化（`:401-403`）；`controller.py:278` `anchor_recognition()` 在 live 路徑有呼叫 | **§1.5 問題 3 的一半** |
| ⚠️ `timing_marks` 的 `open_duration_ms`／`open_to_first_frame_ms` | ⚠️ **demo 連續輪次恆為 `None`。** `controller.py:235` `if reuse_open_source:` → **不呼叫 `note_timing()`**（原始註解：「the lens is already open from the previous round. Do not touch it, and do not claim an open segment.」——這是刻意的 D2b 設計）；而 `qt_window.py:1026` `starting_over_open_camera = self._mode == self._MODE_RESULT`，operator 按「再辨識一次」即傳 `True`，launcher 是 `--continuous` | ❌ **不能用它們替代 `model_load_ms`／`first_frame_ms`（見下方刪除表修正）** |
| `frames_rejected`／`frames_dropped` | `contracts.py:356-357`，已序列化（`:395-396`） | 整體 usable 率（sampled 295／usable 52 = 17.6%） |
| `score_reset_count`／`interval_skip_count` | **事件已 emit**：`session.py:450`（`score_reset`，含 `reset_reason`／`support_before/after`）、`:491`（`interval_skip`）。⚠️ **但 `_emit_event` 在 `session.py:113` 開頭 `if self._event_sink is None: return`，而 live 路徑沒有任何地方傳入 `event_sink`**（`desktop.py`／`qt_window.py`／`controller.py` 皆無）——**事件現在發出就丟** | **D4 §11 第 18b 項缺口** |

⚠️ **r18 那列（`fs=24 fu=3 score=0.6353 margin=0.3811 support_0_of_3`）——3 張可用影格、0 張累積成 support，而最佳幀分數與 margin 都過門檻。這只可能是中間幀觸發了 `score_reset` 或 `interval_skip`，而現有 log 無法分辨是哪一個。** 這就是 D4 18b 的原始歧義，而且它就在手上這 32 列裡。

**必須刪的 v1 提案欄位：**

| 刪除 | 理由 |
|---|---|
| `best_frame_score` | **重複**——`top1_score` 本身就是最佳品質幀的 top1（`session.py:405-412`） |
| `best_frame_index` | 同上 |
| `model_load_ms` | **每次 App 啟動付一次，非每輪**（實測 YuNet 112ms ＋ SFace 54ms）。⚠️ **原理由「已被 open_duration_ms 涵蓋」不成立**——該欄在 demo 路徑恆為 None。**刪除理由是「每輪不適用」** |
| `first_frame_ms` | ⚠️ **原理由「已被 open_to_first_frame_ms 涵蓋」不成立**——該欄在 demo 路徑恆為 None。**若 operator 需要「首次有效幀」的時間，它只能靠 W0-b 的 runbook 記錄，或新增一個不依賴 open 段的計時點——那是新設計,不是搬出既有值。** |
| `support_wait_ms` | 已被 `timing_marks.recognition_duration_ms` 涵蓋 |
| `quality_reject_count` | **重複**——`frames_rejected` 已在 `SessionResult` |
| `no_frame_reason` | **部分重複**——`reason_codes` 已覆蓋那 5 列；缺的只有 11 列零幀（歸 W2） |
| `score_p50`／`score_max` | ❌ **刪除（v8）** —— 這是本專案第三次「存在但無意義的欄位」。⚠️ **即使接線成功，它也答不了問題 2**：逐幀分數只在品質門通過的影格填（`frame_pipeline.py:519`／`:522` 回傳 `identity_scores={}`，`:538` 註解「Only when single face AND quality accepted」），以 32 列計**僅 52/295 = 17.6%** 會帶分數；**且那 52 張全部來自 matched 輪次**（matched 是唯一能累積到 `required_support=3` 的路徑）→ **倖存者偏差，決定分數的裝置條件正是被篩掉的那個。**<br>⚠️ **而 W4 讀的是 CSV——若不寫進 CSV 它算不出來；若寫進 CSV 它就是第三個「有欄位但推論不出結論」的欄位。** 保留它等於製造 §8 已經在防的同一種問題。 |
| `label_identity_mismatch` | **不需要新欄位**——用既有 `label_kind` 與 `top1_identity` 在 W4 算即可 |

**淨效果：欄位數不增反減，但 §1.5 的問題 1（零幀成因）與問題 3（時間分解）被回答。⚠️ **問題 2 不被回答**——見 §8 該條的三點承諾統一說明。**

### 3.4 W4：給 agent 的摘要工具

- 讀固定路徑的 demo log，輸出正確率、混淆分類、分數分布摘要、零幀成因統計。
- **必須同時輸出「誤標」與「真誤認」兩種解釋的 k/N**（§1.6），不得替 agent 收斂到一種。
- **必須明確排除被標記矛盾的列，並報告排除筆數**。
- ⚠️ **必須聲明 W0 的前提**：若 log 沒有跨身分輪替，摘要工具在單一身分資料上通過**不代表它在多身分資料上正確**。**這是 v1 驗收條件的一個隱性缺口。**
- **輸出自帶樣本數與排除數，任何比率帶 k/N 分母**（沿用 D5／D6 呈現規格）。
- **不得引入新的門檻語意；工具只做呈現，不做參數建議。**

### 3.5 W5：與 `results.csv` 的對照驗證

共同欄位的分數必須與 research 模式一致。⚠️ **`elapsed_ms` 在兩種模式的語意可能不同（demo 無 recorder 開銷），此點未經實測，不得假設相同。**

---

## 4. ⭐ W3 欄位判準：執行期是否被讀取（**含方法論修正**）

**任何寫入 log 的參數欄位，必須先證明其值在執行期被讀取。無法證明者不得寫入 log，並列為既存假旋鈑。**

⚠️ **⚠️ 判定必須是語義的，不是檔案清單的。**（v2 的清單法已證實不足。）

**規則：逐一判定每個命中是「比較分支」還是「宣告／轉發／序列化」。** 只有比較分支（`x < threshold`、`if flag`）算執行期消費。**這一步是規則的實質部分，不是補充說明** —— `frame_pipeline.py:142` 的 `with_thresholds` 轉發恰好兩層條件都過（值在 `live/` 下、且非 `contracts.py`），**只有「逐一判定比較分支」這一步擋住它**。

**⚠️ 「可轉」的判準：grep 欄位名不足以判定，必須標明證據等級。**

**v4 曾寫「三條件」，其中條件 2（「在決定行為的那一層被讀」）是同義反覆——已廢除。** 反證：

```
timeout_ms               session.py:148/216  self.profile.timeout_ms
                          controller.py:579/603/614/677  self._engine.profile.timeout_ms
                          desktop.py:363/367            self._engine.profile.timeout_ms
min_support_interval_ms  session.py:483     self.profile.min_support_interval_ms
required_support         session.py:512/540 self.profile.required_support
```

**`timeout_ms` 走 `self._engine.profile.X` 兩層屬性，與 `max_frames`／`sample_interval_ms` 的形式完全一樣——但前者顯然可轉。** 所以「在 `SessionEngine` 裡讀」與「不在 `SessionEngine` 裡讀」**都不構成判準**。

**真正的判準是「該欄位有沒有進入 live 輪次的實際控制流」——這需要呼叫圖追蹤或實測，不是 grep。**

⚠️ **且 grep 欄位名這方法本身不可靠**：單層屬性正則（`\.timeout_ms\b`）只匹配 `self.profile.timeout_ms`，**會漏掉 `self._engine.profile.timeout_ms` 兩層寫法**——這曾讓一份查證差點得出「timeout 不可轉」的結論。

**所以本表改用證據等級，不是一個判定：**

⚠️ **「需實測」不是一個狀態，是一個待辦 —— 本表不設該等級。** 本輪兩位 reviewer 已用執行期實測（真 `LiveController` ＋ `FakeCapture`，改 profile 值再讀回有效值）把所有參數跑完。

| 等級 | 意義 | 例 |
|---|---|---|
| **可轉（控制流已追蹤）** | 已確認該值進入 live 輪次的行為決策 | `required_support`、`min_support_interval_ms`、`timeout_ms`、三個門檻、`review_threshold` |
| **條件式可轉（依執行模式）** | 某些執行路徑可轉、某些不可 | `max_frames`（fixed mode 可轉／demo 不可轉，且 demo 調低會 ValueError） |

**為什麼不能用檔案清單** —— 三個已證實的失效方向：

1. **`contracts.py`** 每個參數都出現多次（欄位宣告、`__post_init__` 驗證、digest 序列化），**全部是建構期**。`sample_interval_ms` 在此出現 7 處、runtime 0 處。
2. ⚠️ **⚠️ `policy/identify.py` 是死碼，不可作為例外條款的依據。** 實測：`identify()` 在 `src/` 內**無任何呼叫端**（只有 `cosine_score` 被 import，live 路徑的門檻判定在 `session.py:435-439`）。**它正是「清單法正確排除」的對象**——不在 `live/` 下、確實不在執行期路徑。**v5 曾為它新增 `policy/` 白名單例外，那是把規則擴張去容納一個死碼檔案，恰好削弱規則自身。**（本條已刪除該例外。）
3. **清單裡的命中可能不是判定點。** `frame_pipeline.py:142-150` 的命中全是 `with_thresholds` 的**轉發**，不是比較分支。**命中數不是證據，判定點位置才是。**

⚠️ **⚠️ 排除清單本身會過期（兩份 review 一致指出）。** 靠「列舉哪些檔案不算執行期」是**負向規則**，新增 `live/` 檔案時必然失準。

**改為正向規則：**
1. **值出現在 `src/facecore/live/` 下、且不在 `contracts.py` → 才可能是執行期消費**（再逐一判定是否比較分支）。
3. **新增 `live/` 檔案或 `policy/` 模組時，本表必須重檢。**

⚠️ **第八個假旋鈑：`quality_policy_version`。** 實測：`live/contracts.py:75` 宣告為 **`str`**，但 `contracts/policy.py:34` 的 `frozen_v1()` 硬編碼為 **`int` = 1**，`with_thresholds()`／`with_detector_gate()` 原樣轉發、**無任何比較或分支**；`live/` 與 `cli.py` 內**零消費點**。**改它對執行期行為零影響，且型別不一致本身就無法接線。**

⚠️ **已知的無名假旋鈑（本輪補上）**：**七個品質門**（`quality.py:32-51` 的 `quality_detector_confidence`／`quality_face_too_small`／`quality_blurry`／`quality_exposure`／`quality_pose_yaw`／`quality_pose_pitch`／`quality_occluded`）code-frozen 在 `PolicyProfile.frozen_v1()`；`profiles/g3-v1.json` **無對應欄位**（只有 `quality_policy_version`），**operator 無法調整**。**它們接受設定、算進 version，卻無 plumbing —— 與 `sample_interval_ms` 同族。**

⚠️ **但與 `max_frames` 不同族**（`max_frames` 是條件式可轉且 by design）。

⚠️ **⚠️ 對照：哪些旋鈑 operator 真的能轉？** 這是 operator 需要一份確切答案的清單，**v3 尚未逐項核對**：

| 參數 | 在 `profiles/` 檔內 | 讀取點 | **證據等級** |
|---|---|---|---|
| `required_support: 3` | ✅ | `session.py:512/540/630/645` `self.profile.required_support` | **可轉（控制流已追蹤）** |
| `min_support_interval_ms: 200` | ✅ | `session.py:483` | **可轉（控制流已追蹤）** |
| `timeout_ms: 5000` | ✅ | `session.py:148/216`、`controller.py:579/603/614/677`、`desktop.py:363/367` | **可轉（控制流已追蹤）** —— 且是「中位 15 秒花在哪」的直接旋鈑 |
| `match_threshold` / `margin_threshold` | ✅ | `session.py:435-438` 比較分支 | **可轉（控制流已追蹤）**（但須走 W0-b 的聯合條件才有意義） |
| `review_threshold` | ✅ | `session.py:781`（經 `compute_baseline_best_quality`）**僅 baseline 分類路徑** | **可轉（控制流已追蹤）** —— 影響鏈已追到底（§7 premise 4 已關閉）；⚠️ 但僅作用於 baseline 分支 |
| **`max_frames: 26`** | ✅ | `controller.py:119` `profile.max_frames if fixed_seconds else max_frames` | **條件式可轉**（見下） |
| **`sample_interval_ms: 200`** | ✅ | `controller.py:57` `sample_interval_ns: int = 200_000_000` **寫死預設**，與 profile 無 plumbing | **不可轉（未接線）** |
| 七個品質門 | ❌ **不在檔內** | `quality.py:32-51` code-frozen；`quality_policy_version` 在 `src/` 內**零讀取點**；且 `ResearchProfile` 是 `str`、`PolicyProfile` 是 `int`，**型別不一致本身無法接線** | **不可轉（code-frozen）** |

⚠️ **`max_frames` 是「條件式可轉」——執行期實測已做（reviewer2 用真 `LiveController` ＋ `FakeCapture` 建構，讀回有效值）：**

```
fixed_seconds=False（demo 路徑）:
   profile.mf=26  -> eff.max_frames=25
   profile.mf=50  -> eff.max_frames=25    ← 改 profile，有效值不動
   profile.mf=14  -> ValueError: max_frames (14) cannot cover the full deadline window
                     （timeout_ms=5000, sample_interval_ms=200 require at least 26 frames）← 調低是「建構失敗」
fixed_seconds=True:
   profile.mf=26  -> eff.max_frames=26
   profile.mf=50  -> eff.max_frames=50
   profile.mf=101 -> eff.max_frames=101   ← 該路徑可轉
```

**所以答案是三分的：demo 路徑不可轉且調低會使 App 建構失敗；fixed mode 可轉。**

⚠️ **`controller.py:58-60` 的原始註解支持這個分類** ——「Ruling: max_frames default kept at 25 by design (task -21/-24 ruling). In fixed mode profile.max_frames wins; in non-fixed mode 25 is a valid execution-layer clamp pinned by test_execution_max_frames_cap_enforced.」** 是 by design，不是失效。

⚠️ **且它不只是不可轉——它還「間接約束」其他參數**（調低會觸發建構期 `ValueError`），**而 launcher 是 `--continuous`、無 `--fixed-seconds`，所以 fixed mode 在 demo 路徑不可達。**

**所以它不是「未接線」或「假旋鈑」，是**刻意設計的條件分支**——demo 路徑（非 fixed mode）走 25 是 by design，且有測試釘住。**它是否「可轉」取決於 operator 要不要用 fixed mode，那是操作模式的選擇，不是 profile 欄位的失效。** 這個區分 v4 未做。

**⚠️ 此表必須在 v3 review 逐項核對後才定稿**——「在 profile 檔內」不等於「可轉」。
⚠️ **另有 grep 本身失效的情形**：`getattr` 可讓搜尋落空（`qt_window.py:641-642` 就在用 `getattr(profile, "review_threshold", 0.30)`）。**語義判定必須人工閱讀命中上下文,不能只靠數命中數。**

### 逐欄位結果（impl 實測，lead 抽驗方法論部分）

| 參數 | runtime 命中 | 判定 | 證據 |
|---|---|---|---|
| `match_threshold` | 5 | ✅ 真旋鈑 | `session.py:437` `top_score < self.profile.match_threshold`（進 `score_reset`） |
| `margin_threshold` | 4 | ✅ 真旋鈑 | `session.py:438` |
| `review_threshold` | 5 | ✅ 真旋鈑（**僅 baseline 分類路徑**） | `session.py:781` `elif top_score >= profile.review_threshold` → `SessionStatus.review`；`qt_window.py:642` 只是顯示用 |
| `required_support` | 11 | ✅ 真旋鈑 | `session.py:512/540` `is_term = len(...) >= self.profile.required_support` |
| `min_support_interval_ms` | 1 | ✅ 真旋鈑 | `session.py:483` → `interval_skip` |
| `timeout_ms` | 8 | ✅ 真旋鈑 | `controller.py:579/603/614` |
| **`max_frames`** | 16 | **條件式可轉**（demo 不可轉／fixed 可轉；demo 調低會 ValueError） | 見上方執行期實測 |
| **`sample_interval_ms`** | **0** | ❌ **假旋鈑** | 全部 4 處在 `contracts.py` 建構期；`controller.py:57` 硬編碼 `200_000_000` ns，與 profile 無任何 plumbing |

**六真兩假——`sample_interval_ms` 是假旋鈑；`max_frames` 是條件式可轉，與它不同族。**

⚠️ **`sample_interval_ms` 特別危險：它接受設定、算進 digest、卻不影響執行。** operator 照直覺去調會得到「改了但沒變」。**這正是 §0 要避免的結果，而 v1 §3.3 正在提議記錄它——若記錄，計畫會主動製造 §0 要防的東西。**

⚠️ **`max_frames` 的佐證方式**：**不要引用「實測 `frames_sampled` 最大 25 吻合」**——lead 抽驗：deadline 5000ms 單獨允許 26 幀，**clamp 與「deadline 恰好切在第 25 幀前」兩種解釋都能產生 25**，該佐證無法區分同值的預測。**改用 plumbing 三段鏈**（launcher 無 flag → 兩處 `DesktopSession` 建構都沒傳 `max_frames=` → 預設 25 生效），該鏈不依賴統計推論。

---

## 4.5 執行順序（不可顛倒）

```
W3（把既有資料接出：timing_marks.recognition_duration_ms、frames_rejected、
    score_reset/interval_skip 計數、probe_kind/presenting_identity、
    event_sink 接到每個 engine）
  └─→ W0-a（runbook 文件，不碰機器）✅ 本輪授權內
        └─→ W1（load_report 三欄接上 log ＋ strict 裁決）
              └─→ W2（11 列 no_frames_captured 的成因分類）
                    └─→ W5（demo vs research 對照驗證）
                          └─→ W4（摘要工具，最後做——它消費 W3 的欄位）

  ⚠️ W0-b（operator 在場執行 W0-a 的 runbook，產出跨身分 ground truth）
     ❌ 不在本輪授權（需 operator 在場）· 必須等 W3 落地
     ⚠️ 它是**唯一能產出跨身分 ground truth 的工作項**，而依 §2 的理由
        （「沒有 W0-b，D7-A 交付的是更精確地測量單一身分的基礎設施，
         而 operator 的核心問題依舊答不出來」），**W0-b 不在鏈中就會被漏看。**
```

⚠️ **§2 與 §4.5 的順序表述已統一為：`W3 → W0-a → W0-b`（W0-b 不在本輪授權），其餘 `W1 → W2 → W5 → W4` 接在其後。**

⚠️ **W3 必須先做**：W0-b 產生的跨身分資料要靠 W3 的 `probe_kind`／`presenting_identity` 才可分析，**所以 W0-b 必須等 W3 落地**（W0-b 需 operator 在場，不在本輪授權內）。

⚠️ **W4 必須最後做**：它消費 W3 的欄位，先做會鎖定錯誤的欄位形狀。

---

## 5. 目標與非目標

### 目標（operator 的實際目標）

- **G1.** App 能開啟相機、建好 gallery、開始辨識。（已可用，本計畫不重做）
- **G2.** 每一輪產生**結構化、可被 agent 可靠讀取**、且**足以回答「零幀成因」與「時間花在哪」；「分數為何集中」需 W3 的逐幀分數接線**的診斷記錄到固定位置。
- **G3.** 改動參數或 gallery 內容後，operator 能**從 log 直接看出差異**，不必人工比對。
- **G4.** log 與 `results.csv` 在共同欄位上一致。
- **G5.（新增）** 診斷 run 有明文的實驗設計（W0），使產出的資料可回答跨身分問題。

### 非目標（本輪明確不做）

- **不做 Android pipeline。**
- **不選產品門檻。**⚠️ 註：本計畫 §1.4 的 non-target sweep **不是門檻選擇**，它是「現況門檻下的裸 FAR 是多少」的量測，且未含 support／品質門聯合模型。**不得被引用為門檻校準結論。**
- **不做模型選擇（D7-B）。**
  ⚠️ **v1 給的理由是錯的，須改寫。** v1 說「等 operator 有跨身分 probe 資料再做」，好像資料還沒蒐集。**實際上：資料已在磁碟上（`辨識組` 13 張、`non-target` 30 張），D6 已經跑過完整的 43 列兩臂比較。真正的阻塞不是「缺 probe 資料」，而是「probe 只有一個身分」。**
  **而 D7-A 的 W1–W5 全部不會改善這點**——再做 32 輪單身分 log，還是單身分。**這正是 W0 存在的原因。**
  **D7-B 併入 D7-A 邊界不乾淨**（W0 的成本在 runbook，log 基礎設施在另一處），故不併入。
- **不修復 `max_frames` 的 demo 路徑脫鉗**（条件式可轉，但 demo 路徑不可達 fixed mode；非 D7-A 範圍），但 W3 不得記錄它。
- **不動 `profiles/g3-v1.json` 的門檻值。**
- **不讀取影像像素、不把照片／crop／embedding 寫進 repo。**

---

## 6. 授權與邊界

**本計畫不授權的事（需 operator 逐項另行 go）**：選定產品門檻、選定或切換模型、任何 Android 端工作、啟動器 `git pull`／`checkout`／`reset`、把任何權重檔／照片／crop／embedding 寫進 repo、**真機量測（須 operator 在場）**。

**技術邊界**：
- ⚠️ **本計畫不得修改 `docs/PROJECT-STATE.md`。** 理由不是「夾帶成本低」，而是**它是跨階段狀態敘述且是 D7-A 新欄位的下游消費者**——`:130` 記載的正是「單一身分不支撐跨身分判別力」，若該檔過期，D7-A 的產出會被餵給一個已錯的狀態敘述。**該檔同步另立 task。**
- `pyproject.toml`／`uv.lock`／`profiles/`／`src/facecore/live/` 的既有語意不得為了本計畫而改。
- W3 記錄的每個參數欄位必須通過 §4 的執行期讀取判準。

---

## 7. 未解 premise（不得假設）

1. **零幀的實際成因**——已證 `timeout` token 不足以分類、成因位置在 `controller.py:369/851`，**未複現相機層故障**。
2. **500 人 gallery 重建耗時**——**只量到 23 人 1.34 s**；線性外推約 29 s／次啟動，**外推本身未量測**。
3. **demo 與 research 模式 `elapsed_ms` 是否同義**——未查。
4. ~~`review_threshold` 在 demo 主路徑的完整影響鏈~~ —— **已追到底（v6 關閉）**：`session.py:629-651` → `compute_baseline_best_quality` → `session.py:779-781` `elif top_score >= profile.review_threshold: return best_frame, SessionStatus.review, None`。**該路徑的輸出已在現有 log 中**（5 列帶 `best_baseline_matched` reason token）。**`review_threshold` 在 demo 路徑確實被讀取，機制是活的。**
5. **non-target 30 張是否代表真實 operator 使用情境**——corpus 是既有產物，代表性未確認。
6. **product 條件下的 FAR**——⚠️ **單張 FAR 已量得**（產品欄：0.363→6.7%、0.5→3.3%、0.55→0.0%）。**但 support 聯合條件量不出來**：corpus 中**每人只有 1 張照片**（實測：`註冊組` 23 張／23 distinct stems、`non-target` 30／30、`辨識組` 13／13），而 `required_support=3` 需要同一輪 3 個不同時間點的影格。**只有真機連續擷取能量得到。**
~~7. `max_frames` 在 fixed mode 的行為~~ —— **已解決（v8 刪除）**：執行期實測已做（`fixed_seconds=True` 下 profile.mf 26→26、50→50、101→101）。**§7 的職責是「未解 premise」，把已解項列入會讓實作者無法判斷該不該依賴它** —— 而 §4.5 的順序論證正建立在「fixed mode 可轉」上。

⚠️ **以上全部需要第一手實測，不得引用本計畫或任何既有報告的敘述作為事實基礎。**

---

## 8. 成功定義（給 review 的驗收依據）

- [ ] 每一輪的診斷記錄寫入**固定位置**（路徑明文指定，不依賴相對路徑）
- [ ] §1.5 的**問題 1（零幀成因）與問題 3（時間分解）能**由 log 回答
- [x] **問題 2（分數跨幀分布）——本輪不做，且理由不是「來不及」。**

⚠️ **① 不能做**：逐幀分數雖在記憶體且已被 `compute_baseline_best_quality` 讀取（`session.py:766-772`），但 `SessionResult` 無對應欄位——**接線在技術上可行**。
⚠️ **② 但做了也沒用**：僅 **52/295 = 17.6%** 的影格會帶分數（`identity_scores` 只在品質門通過時填），**且那 52 張全部來自 matched 輪次**（matched 是唯一能累積到 `required_support=3` 的路徑）→ **倖存者偏差**。
⚠️ **③ 所以正確的表述是**：**本欄位被刪除，而非降級。** 它無法回答問題 2，且 W4 讀不到它。**若未來要回答問題 2，需要的不是一個欄位，而是一套能記錄被品質門與 support 篩掉的影格之分數的設計** —— 屬新設計，須另行授權。

- [ ] **`timing_marks.recognition_duration_ms`、`frames_rejected`、`score_reset`／`interval_skip` 計數出現在 log 中**（既有資料接出，非新設計）
- ⚠️ **`frames_dropped` 不得出現在 log 中** —— 實測 `session.py` 只有 `:88` 初始化與 `:158` 重置，**無任何 `+= 1`**（對照 `frames_rejected` 在 `:293`／`:317` 真的在加）。**搬出一個恆為 0 的欄位是 §4 假旋鈑的同型問題，只是換到資料面。**
- ⚠️ **`open_duration_ms`／`open_to_first_frame_ms` 在 demo 路徑恆為 `None`，若寫入 log 會是永遠為 0 的欄位** —— 若要記錄 open 段時間，**需要新的計時點設計**（因 D2b 刻意不主張 open 段），不得以「搬出既有值」為由寫入
- [ ] **`expected_count`／`loaded_count`／`gallery_rejected` 出現在每一輪**，`gallery_rejected` 含檔名與原因
- [ ] ⚠️ **`event_sink` 已接進**每個** `DesktopSession` 的 engine**（`cli.py:1108` 的第 1 輪 ＋ `cli.py:1427` 的第 2 輪起，**兩處都要**），且 `score_reset_count`／`interval_skip_count` 在**多輪**上非為 0（不是只有第 1 輪）

⚠️ **⚠️ 這是 v5 的錯誤驗收，會製造假資料**：`cli.py:1108` 建構的 engine **有** `event_sink`，`cli.py:1427` 建構的**沒有**。launcher 是 `--continuous`（無 `--fixed-seconds`），**所以 32 列中只有第 1 輪有事件、第 2～32 輪的計數恆為 0**。若驗收只要求「至少一列非為 0」，會由第 1 輪滿足，**而其餘 31 列是 `frames_dropped` 同型的假資料**。
- [ ] **`sample_interval_ms` 與 `max_frames` 未出現在 log 中**
- [ ] **W0 的實驗設計寫入計畫**：跨身分輪替、non-target 輪替、明碼標示呈現者
- [ ] log 記錄 `probe_kind` 與 `presenting_identity`，使 W0 的資料可分析
- [ ] 摘要工具對現有 32 列的輸出**明確指出 34% 零幀與 50% 無 usable frame**，且比率帶 k/N 分母
- [ ] 摘要工具對 4 列矛盾**同時輸出「誤標」與「真誤認」兩種解釋的 k/N**
- [ ] 新欄位在 CI 有實際 passed 計數
- [ ] `pyproject.toml`／`uv.lock`／`profiles/`／既有 23 欄／`PROJECT-STATE.md` 零變更
- [ ] 既有 store 與來源照片前後 hash 不變

---

## 教訓段：本規劃收斂過程中的兩個同型失效

**⚠️ 兩次都是「到達一個狀態就當完成」，不驗證下游是否真的接續。**

### ① 計畫層（本 session）：修正完成 ≠ 派了下一輪 review
v2 兩份 review 同時 REJECTED（3 ＋ 8 項），我專注於逐條套用修正並驗證，**但「修正完成 → 派下一輪 review」這一步沒有組出來，停了 40 分鐘**。由 commander 發現並指出。

### ② 產品層（D7-B 線）：單幀達標即結束，不確認後續幀
`best_baseline_matched` 這個 reason code 意味著：**單幀已達 matched 帶，系統就結束了，沒有去確認後續幀能否滿足 `required_support=3`。** 5 列分數 0.5621～0.6995（遠高於門檻）全部栽在這裡。

**⚠️ 兩次是同一個反應模式：把「完成了步驟 A」當成「A 之後的事也會發生」。** 一次在計畫流程、一次在產品控制流。

**這一條的實務含義**：任何「做完就會自然接上」的假設都必須顯式驗證下游，**因為本專案已有兩個獨立實例**（無論它們是否相關）。

### ③ 第三個同型：判定方法本身失效
本規劃建立「W3 只記錄執行期被讀的參數」這條規則時，**用 grep 欄位名作為判定手段**。實測發現單層屬性正則會漏掉 `self._engine.profile.X` 兩層寫法——**這個方法本身曾導致一份獨立查證得出與事實相反的結論**。

**三者同源：都以「看起來已經完成／正確」的代理量代替對下游實際狀態的驗證。**

---

## 附：本修訂推翻的 v1 判斷（供 review 對照）

| v1 判斷 | 處置 |
|---|---|
| §1.1「24 欄」 | ❌ 錯，實際 **23** |
| §1.3「中位數 15205.3」 | ❌ 錯，實際 **15182.2** |
| §0／§1.4「沒有可調的參數空間」 | ❌ **推翻**——門檻在 non-target 上有明確空間；真 knob 是 `required_support`／`min_support_interval_ms` |
| §1.4「窄分布 → 500 人可分性不足」 | ❌ **無資料支持**——0.62–0.73 是單一身分的輪間變異 |
| §1.2「34% 完全沒有畫面……相機層失敗，不可改善」 | ⚠️ **需收斂**：v1 的標題明確限於 11 列零幀，**v1 從未把 16% 歸入任何分類**。v1 的錯誤是「以 34% 代替 50% 描述無 usable frame」，不是「誤分類 16%」。**而那 16%（品質門過濾）確實無旋鈑——與 v1 的結論方向一致。** |
| §1.6「是 operator 按錯標記」 | ❌ **未證**——真誤認同樣可能，且該門檻下誤認是預期行為 |
| §1.7「可能靜默進入 gallery」 | ❌ 錯——檢查完整；真正的問題是 `strict=False` 的**靜默排除**，且 UI 已顯示但**與 log 脫鉤** |
| §3.3 的 15 欄提案 | ⚠️ **過度設計**——多欄位重複或已在 repo 中算好 |
| §3.3 含 `sample_interval_ms` | ❌ **假旋鈑**，記錄它會製造 §0 要防的結果 |
| §2「不做模型選擇」 | ✅ 判斷對，❌ **理由錯**（不是缺資料，是 probe 單一身分） |
| §1.5 三問題的欄位需求 | ⚠️ **成本高估**——`TimingMarks` 已在 live 路徑被填，只是沒寫出 |
| D0 §11「13 項」 | ❌ 實際 **19 項**（`docs/g3-local-test-sop.md` 表格列數）。⚠️ **「其中 8 項待驗」無法從 repo 驗證**（屬 task/dispatch 層），故不在本計畫凍結該計數。 |
