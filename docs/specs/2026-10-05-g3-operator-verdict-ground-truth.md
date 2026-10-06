# G3-w：operator 結果標註與 ground truth 自動帶出（spec）

- **base full HEAD**：`a860ee09313858b5fd3e5b612f76daf9c46f3d87`（spec review 時）→ ⚠️ **落地時以 `12f0bd6`（#156 merge）為 base**，因為 §2.3 的比對是針對該 commit 的 `runbook`
- **建議落點**：`docs/specs/2026-10-05-g3-operator-verdict-ground-truth.md`
- **landing mode**：⚠️ **獨立 spec-only PR，且該 PR 必須同時更新 `runbook`** —— 見下方「落地時的連帶範圍」
- **授權**：operator 2026-10-05 TUI 直接指示（「先派 review spec，你跟 lead 得到 spec 共識落地後，可以派 lead 開始執行 pr loop」）；裁決記於 `d-20261005003405606386-2`
- **supersedes**：`runbook` 的 `nontarget` 側「值域待 W0-b 排程定義」掛起態（§2.3 裁決該值域）
- **spec review**：兩輪，`t-20261005014011506028-98120-19`（lead），裁決歷程見 §8.4

## ⚠️ 落地時的連帶範圍（不可拆開）

⚠️ **本 spec 落地 PR 必須同時改 `docs/w0a-diagnostic-run-runbook.md`**，因為 §2.3 已決定 `nontarget` 側的值域（`outsider`）。若只落 spec 而不動 runbook：

- 在「W0-b 排程時的記錄規則」表中 `presenting_identity` 那列仍寫「這欄填什麼必須在 W0-b 排程時定義」⚠️ **而 spec 已定義** → 兩份權威文件矛盾
- 同表 `填不進去怎麼辦` 那列仍寫「`nontarget` 側的 `presenting_identity` **目前仍填不進去**，因為它的值域還沒定義」⚠️ **而 spec 已定義** → 實作者會讀到過期狀態
- 「可達組合」表（`### 現在的狀態是分側的` 子節內）列了四種組合⚠️ **其中兩種在 spec 實作後不再可能出現**（見 §5.3）

⚠️ **⚠️ 但「可達組合」表（`### 現在的狀態是分側的` 子節內）的更新時機是「實作 PR」** —— ⚠️ **因為 spec 落地 ≠ 程式已改**，⚠️ **spec 落地後那些組合仍然存在（程式還沒改）。** ⚠️ **所以落地 PR 只更新「值域已決定」這件事，表格留到實作 PR。**

---

## 1. 問題陳述

### 1.1 現況

W0-b 的兩欄 ground truth（`probe_kind`／`presenting_identity`）在 `#154`（merged）拿到輸入路徑：Qt 下拉 ＋ 身分選單（`src/facecore/live/qt_window.py:422-470`）。**但那個設計要求 operator 在每一輪手動選兩個下拉。** operator 明確拒絕：

> **不該由 operator 手動填，`probe_kind` 跟 `presenting_identity` 改成程式自動判定。**

⚠️ 但那兩欄的**語意**（`runbook:516-519`）決定了「自動判定」的可行邊界：

| 欄位 | 記什麼（runbook 逐字） |
| --- | --- |
| `probe_kind` | 這一輪是「target」（測試者有註冊）還是「nontarget」（測試者沒註冊） |
| `presenting_identity` | 這一輪鏡頭前的人是誰（例如 `enroll-07`） |

> **有這兩欄，才能把「系統的判斷」和「實際情況」對起來** —— 這是回答「會不會認錯人」的前提。

**這兩欄在定義上就是 operator 的主觀觀察。** 程式端能拿到的獨立訊號只有兩類，**兩者都不含「鏡頭前是誰」**：

1. `gallery.embeddings` 的鍵 —— 註冊組有哪些人，不含當前是誰
2. `desktop.display_identity()`（`src/facecore/live/desktop.py:331`）—— **「系統認成了誰」**

⚠️ **用 (2) 自動填 `presenting_identity` 會造成循環**：`display_identity()` 回傳 `matched_identity`，即系統判斷本身。runbook 說明這兩欄存在的理由正是要把「系統的判斷」和「實際情況」對起來 —— **若 ground truth 來自系統判斷，那個 run 在方法論上無效，而且看起來會完全正常。** 這比欄位空著危險得多：空著是誠實狀態，自動填錯欄位是**假的 ground truth**，會污染跨身分統計。

### 1.2 本 spec 的解法

⚠️ **不追求「全自動推導 ground truth」**（§1.1 已證不可能）。改為**把「每輪手動選兩個下拉」縮成「按一個按鈕 ＋ 出錯時才問一次」**，並讓程式**從 operator 已經在做的動作**（按對／按錯）**推導能推導的那一半**。

operator 原話：

> 「UI 怎麼設計，你們決定。」

---

## 2. Operator 的 2×2 模型與五格

operator 2026-10-05 定義：

> 「有找到此人」「沒找到此人」→ 各有「結果正確」「結果錯誤」
> 「有找到此人」→ 錯誤時再分：**此人有在註冊組**／**沒有註冊**
> 「沒找到此人」→ 錯誤 = **此人有註冊，但沒找到**
> （沒找到／錯誤 這種情況「因為寧可錯殺一人，不可錯放一人的嚴謹狀態，本來就有可能發生」）

展開成五格：

| # | 系統結果 | operator 判定 | 實際是誰 | `probe_kind` | `presenting_identity` | 要 operator 輸入嗎 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `matched`（認成 enroll-07） | 正確 | enroll-07 | `target` | 空（系統已認對） | **不用** |
| 2 | `matched` | 錯誤，**此人**在**註冊組 | 應為 enroll-23 | `target` | `enroll-23` | **要**（註冊組下拉） |
| 3 | `matched` | 錯誤，**此人不在**註冊組 | 隨機路人 | `nontarget` | ⚠️ **`outsider`**（或清單檔名，見 §2.3） | **要**（選 nontarget 側） |
| 4 | `unknown`／`review`／`timeout`（沒認出） | 正確 | 沒註冊的人 | `nontarget` | `outsider` | **不用** |
| 5 | `unknown`／`review`／`timeout` | 錯誤 | 應為 enroll-23 | `target` | `enroll-23` | **要**（註冊組下拉） |

⚠️ **格子 5 是門檻政策「寧可錯殺不可錯放」真正要量的東西** —— registered 被拒的比例。30 張 non-target 只能量誤放（格子 3），量不到誤殺。**兩邊都要有數據才談得上調門檻。**

⚠️ **格子 4／5 的「沒認出」橫跨三個實體值**（`unknown`／`review`／`timeout`），⚠️ **不是單一值** —— 詳見 §4.3。⚠️ **`not_found` 這個值在程式裡不存在**（spec 初版的錯誤，已修正）。

⚠️ **`presenting_identity` 在新 UI 下不再有「刻意留空」的情形** —— 見 §2.3。⚠️ **這是與 `runbook`（#156 合併）比對後的修正**，初版曾設計成留空。

### 2.1 格子 3 的身分從哪來

operator 2026-10-05 裁決（明確）：

> 「(b) 只是知道『這人沒註冊』，但對不上 30 張裡哪一張」

⚠️ 真人測試時鏡頭前是真人，**對不上 30 張裡哪一張**。那 30 張 `nontarget-01.jpg`…`nontarget-30.jpg` 是**照片**，不是真人對照表。

⚠️ **但這不犧牲 W0-b 的指標**：`probe_kind=nontarget` 照樣記錄 → **誤放率仍然量得到**（那正是格子 3 的定義）。失去的只有「哪一個 non-target 最容易被誤認」這個歸因維度 —— ⚠️ **除非 operator 手邊有對照表**，那時 §4.4 的 non-target 清單就派上用場。

### 2.2 格子 3 的 non-target 清單仍然有用（自動讀）

⚠️ **即使真人測試用不到，`~/Downloads/face_sample/non-target/` 的檔名清單在「拿著對照表」的情境仍然有用**（例如 operator 手邊有同批人的對照表）。operator 要求：

> 「那 30 張清單要程式讀，未來我可能會變換 non-target 的資料，所以**一切都要自動化，不能每次都要改程式碼**。」

→ §4.4 規定：程式讀資料夾，**不得寫死任何檔名**。


### 2.3 ⚠️ `presenting_identity` 不再留空 —— 與 `runbook`（#156）比對後的修正

⚠️ **本節是 spec 落地前與已合併 `runbook` 交叉比對的結果，解決了一個實質衝突。**

⚠️ **衝突**：`#156`（merged `12f0bd6`）剛把「W0-b 排程時的記錄規則」表的 `presenting_identity` 列寫成：

> `presenting_identity`｜ 測試者在註冊組裡就填其身分；**沒有註冊的測試者，這欄填什麼必須在 W0-b 排程時定義**（例如 **`outsider`** 之類的標記），**不可留空**

⚠️ **而本 spec 初版設計格子 3／4 的 `presenting_identity` 留空** —— ⚠️ **直接違反 `runbook` 的「不可留空」。** 兩份權威文件會互相矛盾，⚠️ **而實作者會各自猜。**

⚠️ **裁決：採用 `runbook` 的方向 —— `nontarget` 側填 `outsider`，不留空。**

⚠️ **理由**：同表「`probe_kind` 的合法值」列說「**沒有第三個值**，也不准空著假裝有值」，空字串的語意是「**沒有記錄**」（`:522`）⚠️ **而格子 3／4 是「已記錄、已知的非目標輪」** —— **兩者語意不同，不該共用空字串。**

⚠️ **`outsider` 是該列舉的例，不是硬性要求** —— ⚠️ **但它必須是 `runbook` 已定義過的標記**，⚠️ **不可由實作者自創**（自創等於「`probe_kind` 的合法值」列的「沒有第三個值」）。

⚠️ **該列說這是「W0-b 排程時要決定的」，而 `填不進去怎麼辦` 那列明說 `nontarget` 側的值域「是 W0-b 排程要決定的，不是 W0-a 的缺口」** ⚠️ **所以本 spec 落地時就是那個「決定」的時刻** ⚠️ **且需同步更新 `runbook`，說明值域已被決定、`#154` 當年刻意 disabled 的理由已消失。**

⚠️ **實作上的重要後果**：`#154` 的 `_populate_presenting_identities`（`qt_window.py:560-573`）**刻意讓 `nontarget` 側 disabled**（註解說「待 W0-b 排程定義」）⚠️ **本 spec 落地後那個 disabled 理由消失**，`nontarget` 側必須可填 `outsider`／清單檔名。

⚠️ **⚠️ 這也影響 §5.3 的值域設計** —— ⚠️ **該節記錄了 `operator_verdict` 各種狀態如何被區分，⚠️ 以及「刻意略過」為何最終採用明確值 `skipped` 而非空值** —— 見該節。

---

## 3. 「無效輪」也要分正確／不正確

operator 2026-10-05 指示：

> 「『無效輪』沒找到人臉也要分結果正確、結果不正確，因為可能是有臉判成無臉、無臉判成有臉」

⚠️ **operator 同意本 spec 的一項收窄**（2026-10-05 TUI「1.2. 照你建議」）：

**`invalid_input`（沒偵測到人臉）那一輪只問正誤、不問是誰。**

理由：那一輪沒有可填的身分（系統根本沒進入比對）。但「不正確」這個標記本身**就是漏檢率**（有臉判成無臉），對門檻調校有價值。

### 3.1 ⚠️ 一個必須記錄的資料模型限制

operator 提到的兩種情況裡，**「無臉判成有臉」不會產生 `invalid_input`**：

| 情況 | 現況 `result` 記成 | 本 spec 的處理 |
| --- | --- | --- |
| 鏡頭前**沒有**人臉 → 判無臉 → 正確 | `invalid_input` + `correct` | ✅ |
| 鏡頭前**有**人臉 → 判成無臉 → **不正確** | `invalid_input` + `incorrect` | ✅ 靠 `operator_verdict` 區分 |
| 鏡頭前**有**人臉、辨識成功，但**此人其實沒註冊** | `matched` + `incorrect` + `nontarget` | ✅ 這就是格子 3 |

⚠️ **但「無臉判成有臉」這個分支在程式裡不存在** —— 實測 `session.py:852-859`：`matched` 的前提是 `face_count == 1`，**「無臉」在定義上不可能產生 `matched`**。該情況的正確落點是**格子 3**（系統認成某人，但該人未註冊）。

⚠️ **即：「無臉判成有臉」= 格子 3，不是第六格。** 系統「認成某人」而該人其實未註冊，記錄形態與格子 3 完全相同。**五格模型已覆蓋 operator 的兩種情況，無需第六格。**

⚠️ **上一句「不會產生 `invalid_input`」措辭不精確，本節已修正**（原意是「不會產生 `invalid_input` 那個『沒偵測到人臉』的分支」）。

---

## 4. UI 設計（commander 決定，operator 已授權）

### 4.1 移除兩個每輪下拉

`qt_window.py:446-470` 的 `probe_kind_combo` ＋ `presenting_identity_combo`（`#154` 加入）**從標註流程移除**。

⚠️ **但 CSV 兩欄保留、`writer_wiring` 三條守護保留** —— 兩欄仍被轉發，只是值的來源改變。

### 4.2 每輪結束的標註區

```
辨識結果：enroll-07（0.5226）　邊際 0.2021
┌──────────────────┬──────────────────┬──────────┐
│   ✓ 結果正確      │   ✗ 結果錯誤      │  略過    │
└──────────────────┴──────────────────┴──────────┘
```

- 按「結果正確」→ 直接下一輪，**零輸入**。
- 按「結果錯誤」→ **下方才展開輸入區**（progressive disclosure：按了才知道要填什麼）。
- ⚠️ **按「略過」→ operator 選擇不判這一輪，直接下一輪** —— ⚠️ **記 `operator_verdict=skipped`**（§5.1），⚠️ **這是與 ✓／✗ 並列的第三個動作，不是「忘了按」的同義詞**。

⚠️ **「略過」放與 ✓／✗ 同列，理由**：它是**與判定同層級的 operator 選擇**，不是輸入區裡的一個欄位。⚠️ **放在同列才能讓它在「有判斷結果」的標註流程中可用** —— ⚠️ **而若放進「結果錯誤」的展開區，它只在 operator 已選錯之後才出現，⚠️ 那就不是「不判這一輪」，而是「判錯了再補記」，語意不同。**

⚠️ **⚠️ 「略過」不在 `invalid_input` 輪出現 —— §4.5 的兩按鈕例外維持不變** ⚠️ **`invalid_input`（沒偵測到人臉）沒有可判斷的辨識結果：operator 能說的只有「有沒有漏了人」這個旗標，⚠️ 而那不是一個身分，也不是一次對錯判定。** ⚠️ **所以那一輪沿用 §4.5 的 ✓／✗ 兩按鈕，不提供「略過」** —— ⚠️ **理由與 §4.6 的 `cancelled`／`error` 相同：operator 沒有東西可判，略過按鈕只會誘使他亂按。**

⚠️ **⚠️ 所以「略過」的可用範圍是「有判斷結果的標註流程」** —— ⚠️ **`matched`／`unknown`／`review`／`timeout`（§4.3）可用；⚠️ `invalid_input` 不可用（§4.5）；⚠️ `cancelled`／`error` 不可用（§4.6，根本不顯示標註按鈕）。**

⚠️ **⚠️ 「略過」按下去之後 `probe_kind` 與 `presenting_identity` 要怎麼處理，本 spec 未定義** —— ⚠️ **是仍記錄推導值、還是兩欄留空，是實作決定，此處不假定。** ⚠️ **⚠️ 這是本 spec 刻意留下的未解問題：實作時必須回報，不得自行假設。**

⚠️ **「正確」的格子仍然要寫 `probe_kind`** —— 程式從系統結果推導：認成某人 → `target`；沒找到 → `nontarget`。⚠️ **這是推導而非觀察，所以「沒找到 → `nontarget`」在格子 5 會錯**，而那正是 operator 按「結果錯誤」要改的情況。**推導值一律可被 operator 覆寫。**

### 4.3 展開區隨系統結果變化

⚠️ **系統結果的實體值域**（`SessionStatus`，`src/facecore/live/contracts.py`）：`matched`／`review`／`unknown`／`invalid_input`／`timeout`／`cancelled`／`error`。

⚠️ **沒有 `not_found` 這個值。** 「沒找到此人」在程式裡是 **`unknown`**（`session.py:866`、`:875`、`:885`）。本 spec 一律用實體值。

⚠️ **但「沒找到」實際橫跨三個值**，operator 看到的樣子不同：

| 實體值 | 觸發條件 | UI 顯示名字？ | operator 的感受 |
| --- | --- | --- | --- |
| `unknown` | 分數低於 `review_threshold`（`session.py:885`）；或無 `identity_scores`（`:866`）；或無 runner-up margin（`:875`） | ❌ 沒有 | 「沒找到此人」 |
| `review` | 分數介於兩個門檻之間（`session.py:883`） | ❌ **沒有**（`display_identity()` 只在 `matched` 回傳名字） | 「沒找到此人」（但系統其實有候選人，只是不確定） |
| `timeout` | 有 `review`／`matched` 前景但證據段不足（`session.py:741`） | ❌ 沒有 | 「沒找到此人」 |

⚠️ **`review` 對 operator 而言看起來跟 `unknown` 一樣**（都不顯示名字），但它**內部有 top1/top2 與分數** —— 那是「系統其實很接近答對了」。⚠️ **這是門檻調校最有價值的區間，不該跟 `unknown` 混為一談。**

⚠️ `cancelled`（`:678`）與 `error`（`:792`）是**操作失敗，不是判斷結果** —— 不進本流程（見 §4.6）。

**因此標註區有三種形態：**

⚠️ **⚠️⚠️ 下表是 P3 要實作的目標狀態，不是現況描述** ⚠️ **⚠️ 現行 source 沒有「`invalid_input` 輪把 `probe_kind` 清空」這個機制** —— ⚠️ `_probe_kind_value`／`_presenting_identity_value`（`qt_window.py`）完全對 status 盲目，只讀 combo 的目前選取值，⚠️ 而 `enter_ready` 與每輪流程都不重設那兩個 combo。** ⚠️ **⚠️ 所以實際行為是：operator 選過一次之後，之後每個 `invalid_input` 輪都會寫出那個殘留值 —— 下表 `invalid_input` 那一列的「自動帶出」格描述的是 P3 必須建立的行為，不是現在會發生的事。**

⚠️ **⚠️ 那個殘留問題該怎麼處置（隨 status 清空／保留上一輪值／明確寫空）是 P3 的設計裁量，本 spec 不決定。** ⚠️ **⚠️ 但它必須被裁決**，⚠️ 否則 §5.3 表格的 `invalid_input` 列與 §4.2「`invalid_input` 不提供略過」的判定都建立在「那一輪 `probe_kind` 是空的」這個前提上，⚠️ 而該前提在現行 source 不成立。

| 系統結果 | 問什麼 | 選項 | 自動帶出（⚠️ **目標狀態**） |
| --- | --- | --- | --- |
| `matched`（認成某人） | 實際是誰？ | ① 註冊組下拉（gallery keys）<br>② non-target 清單下拉（§4.4）<br>③ **`outsider`（隨機路人，不在以上）** | ① 填身分 ＋ `probe_kind=target`<br>② 填檔名 ＋ `probe_kind=nontarget`<br>③ 填 `outsider` ＋ `probe_kind=nontarget` |
| `unknown`／`review`／`timeout` | 實際有沒有人註冊？ | ① 註冊組下拉<br>② **`outsider`（沒有）** | ① 填身分 ＋ `probe_kind=target`<br>② 填 `outsider` ＋ `probe_kind=nontarget` |
| `invalid_input` | **不問是誰** | — | `probe_kind` 空（⚠️ **P3 目標狀態；現行 source 會殘留上一輪值，⚠️ 處置未裁決**） |

⚠️ **選項②／③ 全部填值，沒有「留空」選項** —— 見 §2.3，⚠️ **這是與 `runbook` 該列「不可留空」對齊的結果**。

⚠️ **`review` 與 `timeout` 顯示時要額外標明系統狀態**（例如「接近門檻但證據不足」），⚠️ **因為那個區間正是門檻該調的地方** —— 讓 operator 知道他這輪不是「完全沒認出」。

⚠️ **格子 4／5 的量測口徑**（影響誤殺率計算，lead 指出這是需要裁的）：

⚠️ **裁決：格子 4（正確拒絕）與格子 5（誤殺）涵蓋 `unknown`／`review`／`timeout` 三值。** 理由：operator 看到的三者**都是「沒找到此人」**（都不顯示名字），他無法區分。⚠️ **若只算 `unknown`，則 operator 按「結果錯誤」時會看到系統狀態是 `review` 而無對應格子** —— 那是 spec 的漏洞，不是量測的精確。

⚠️ **但 `result` 欄位照實記錄三值**，⚠️ **下游分析必須按 `result` 分組**（誤殺率要能拆成「完全沒認出」與「接近答對」兩個子類）。⚠️ **本 spec 不要求合併成單一數字。**

⚠️ **選項③「隨機路人＝`outsider`」的存在理由**：operator 明確說真人測試時「沒註冊的人沒照片檔案，所以也無從登記」。⚠️ **不可用「讓使用者打免費文字」代替** —— 那是 `#154` 註解（`qt_window.py:429-433`）明確拒絕的形狀：

> A typed identity that is one character off (`enrol-24`) becomes a silent ground-truth error, and these columns exist precisely to be ground truth.

⚠️ **自由文字欄一律不做**，所有選項都要是封閉域。⚠️ **`outsider` 必須是 `runbook` 已定義的固定標記，不得由實作者自創字串。**

### 4.4 non-target 清單：程式讀，零硬編碼

- App 提供「選 non-target 資料夾」路徑（`QFileDialog.getExistingDirectory`）。
- 程式枚舉資料夾內影像檔（副檔名集合見 `src/facecore/live/frame_pipeline.py:162`：`.jpg/.jpeg/.png/.bmp/.tiff/.tif/.webp`），**以檔名 stem 作為清單項**。
- ⚠️ **清單不得寫死任何檔案名，不得寫進 repo。** 換資料 = 換資料夾內容。
- ⚠️ **清單是執行期狀態**（operator 機器上的 repo 外路徑），不是程式常數。這是 `CLAUDE.md` 已立之「per-batch 數字不得進權威文件」規則的精神（operator 2026-10-04 立的規則，`#155` 已把它變成條文）。
- 未選資料夾時，選項②不可用（disabled 且顯示「尚未選 non-target 資料夾」），⚠️ **但選項③ `outsider` 永遠可用** —— 那是不持有對照表時的預設路徑。

### 4.5 無效輪的標註

`invalid_input` 輪**不出現身分輸入**，只出現 ✓／✗ 兩個按鈕。

### 4.6 `cancelled`／`error` 不進標註流程

⚠️ **⚠️⚠️ 本節描述的是 P3 必須建立的機制，現行 source 沒有。** ⚠️ 現行 source 的 `cancelled`／`error` 輪**沒有任何寫入路徑**（實測見下段），⚠️ 而「該輪直接記錄為失敗」是 P3 必須建立的行為，⚠️ **現行 source 做不到這件事** —— ⚠️ **下游不可把本節讀成現況描述。**

⚠️ **⚠️ 若 P3 不建立這個機制，後果是：`result=cancelled`／`error` 的那一列在 CSV 中永遠對不上任何資料，`cancelled` 與 `error` 這兩個 `result` 值將無法從 CSV 區分開來。** ⚠️ **⚠️ 而 §5.3 表格中依賴這個前提的那一列也會永遠對不上資料** —— ⚠️ 見 §5.3 該列的標記。

這兩值代表**操作失敗**（operator 中斷、系統錯誤），**不是辨識判斷的結果**。⚠️ **不顯示標註按鈕**，該輪直接記錄為失敗。

⚠️ **理由**：operator 對這兩種情況沒有「正確／錯誤」可判斷 —— 他沒有看到一個辨識結果。⚠️ 若顯示按鈕，operator 會被迫亂按，污染 `operator_verdict`。

⚠️ **⚠️ 現行機制：`cancelled`／`error` 那一輪也不寫 row。** ⚠️ 實測（AST 枚舉 `qt_window.py` 每個函式的行號區間並搜尋 row-write token）：全域 row-write 只出現在 `_record_unlabeled_round`（`:1034`）與 `_press_key`（`:1687`／`:1706`），而 `cancel_clicked`（`:1245-1270`）與 `_update_terminal`（`:1747-1760`）**零命中**；`_record_unlabeled_round` 的唯一呼叫者是 `recognize_again_clicked`（`:1091`），`_press_key` 的呼叫者只有兩顆標註按鈕。⚠️ record 模式的 `commit_g3_rounds` 在失敗時走 `recorder.abort`，abort 也不寫 row。** ⚠️ **⚠️ 所以「`cancelled` 有列、停止相機沒列」這個對照在現況下不成立** —— ⚠️ 上一版 spec 曾這樣寫，那是**未經量測的斷言，已刪除**。

⚠️ **⚠️ 真正該記的是：§4.6 與 §4.7 在「是否寫 row」上現況相同（都不寫），⚠️ 差別在結束方式與 session 生命週期** —— ⚠️ `cancelled` 是 operator 按 Cancel 走 `cancel_inference()` 結束**那一輪**再回 Ready，session 還在；⚠️ 停止相機是 `stop_camera_clicked` 主動 `detach()`／`close()` **結束整個 session**。

### 4.7 停止相機：⚠️ 現行機制下**不寫 row**

⚠️ **operator 按「停止相機」結束該輪時，那一輪不寫任何 row** —— 不是「寫一列空的」，而是**完全不寫**。

⚠️ **實測依據**：`stop_camera_clicked`（`src/facecore/live/qt_window.py`）走的是關相機 → detach／close → 回 Ready → 清 preview → `_set_status`，⚠️ **其函式體內零次呼叫** `_record_unlabeled_round`／`append_g3_demo_results_csv`／`commit_g3`。⚠️ **所以 §5.3 的表格沒有、也不該有它那一列。**

⚠️ **⚠️ 與 §4.6 的關係：同類，但現況下「都不寫 row」—— 這是本節存在的唯一理由：**

| | §4.6 的 `cancelled`／`error` | 停止相機 |
| --- | --- | --- |
| operator 在做什麼 | 沒在判斷那一輪，他結束**那一輪** | 沒在判斷那一輪，他結束**整個 session** |
| 是否寫 row（**現行實測**） | ⚠️ **不寫**（`cancel_clicked` 零 row-write 呼叫） | ⚠️ **不寫**（`stop_camera_clicked` 零 row-write 呼叫） |
| session 是否還在 | ✅ 還在，回 Ready 可繼續下一輪 | ❌ 已 `detach`／`close`，session 結束 |
| `result` 欄有無該輪 | ❌ 無 | ❌ 無 |

⚠️ **⚠️ 所以「每輪都有 row」這個假設，在 `cancelled`／`error` 與停止相機**兩者**上都錯** —— ⚠️ 真實的坑比只處理停止相機更大。⚠️ **⚠️ 而 §5.3 那張表裡的「該輪未進入標註流程」一列（`probe_kind` 空 ＋ verdict 空），⚠️ 在現行機制下對 `cancelled` 也不成立** —— ⚠️ 那正是本節要提醒 P3 與下游的事：**那張表描述的是新規則下的目標狀態，不是現況。**

⚠️ **⚠️ 注意它與「再次辨識」也不同** —— ⚠️ 「再次辨識」會呼叫 `_record_unlabeled_round` 寫一列（§5.3 的略過入口，記 `skipped`），⚠️ 而停止相機連那一列都沒有。

---

## 5. 資料模型變更

### 5.1 新增 CSV 欄位 `operator_verdict`

| 屬性 | 值 |
| --- | --- |
| 欄名 | `operator_verdict` |
| 值域 | `correct`／`incorrect`／`skipped`／空 |
| 位置 | ⚠️ **追加為最後一欄（index 38）** |
| 理由 | ⚠️ **不得插入中間** —— `G3_DEMO_RESULTS_CSV_COLUMNS` 是 38 欄固定順序的 tuple 契約（`src/facecore/research/cli.py:376-551`），`append_g3_demo_results_csv` 用它當 `DictWriter` 的 `fieldnames`（`:891`），`:874` 會驗 header 一致。插中間會破壞既有欄位順序與 header 驗證。 |

⚠️ **⚠️ 「空」的語意已收斂為「該輪未進入標註流程」** —— ⚠️ **即 §4.6 的 `cancelled`／`error`，或 operator 中途中止。** ⚠️ **「空」不再代表「operator 沒標註」這種籠統說法**，⚠️ **刻意不判的場合一律寫明確值 `skipped`（§4.2 的「略過」動作）。** ⚠️ **⚠️⚠️ 而「空」代表什麼，⚠️ 取決於 P3 是否建立 `cancelled`／`error` 的記錄機制** —— ⚠️ **現行 source 那一輪沒有任何寫入路徑（§4.6），⚠️ 若 P3 不建立，⚠️ 這個語意在 CSV 中永遠沒有對應資料。**

⚠️ **⚠️ 為什麼「刻意略過」必須是明確值而不是空** —— ⚠️ **在舊規則下，「刻意略過」「漏寫」「忘按」三者的 CSV tuple 位元組相同，⚠️ 沒有任何欄位可以區分，⚠️ 所以「空 verdict = 刻意略過」只能是推測。** ⚠️ **新增 `skipped` 之後這三種情況不再是同一個 tuple，⚠️ 碰撞消失，「刻意略過」變成可從值直接讀出的事實。**

⚠️ **⚠️ 一個直接後果：ground truth 有值而 `operator_verdict` 空，成為不可達狀態** —— ⚠️ **任何進入標註流程的輪次都會寫 `correct`／`incorrect`／`skipped` 三者之一，⚠️ 而未進入標註流程的輪次 `probe_kind` 是空的。** ⚠️ **所以「`probe_kind` 有值 ＋ `presenting_identity` 有值 ＋ verdict 空」在新規則下是異常，不是合法的第三種狀態**（見 §5.3）。

⚠️ **追加前的欄位數 = 38**（不是某些文件寫的 35 或 50）—— ⚠️ **⚠️ 這是本 spec 追加 `operator_verdict` 之前的實測值（AST 解析）；#158 已把它追加到最末，所以現行 source 的實測值是 39。** `probe_kind` = index 28、`presenting_identity` = index 29。追加後 = **39 欄**。

### 5.1.1 ⚠️ 追加會打破三條既有守護（本節為 spec review 後新增）

⚠️ **三條既有守護假設了「38 欄」或「末三欄是 W1 三欄」，追加後全部轉紅。** 這三條 spec 初版未列，是 review 缺口。

| # | 守護 | 現況斷言 | 追加後 | 處置 |
| --- | --- | --- | --- | --- |
| **G1** | `tests/cli/test_d7_w1_gallery_visibility.py:69-73` `test_new_columns_are_appended_after_the_existing_35` | `G3_DEMO_RESULTS_CSV_COLUMNS[-3:] == ('expected_count', 'loaded_count', 'gallery_rejected')` | `[-3:]` 變成 `('loaded_count', 'gallery_rejected', 'operator_verdict')` → **紅** | ⚠️ **改寫為 `[-4:]` 斷言（逐字見下）**，同時保住**位置**與**順序** |
| **G2** | `tests/live/test_d7_w0b_probe_kind_input.py:291` | `assert widths == {38}` | 39 → **紅** | 更新為 `== {39}`。⚠️ **該檔同時在重寫範圍內（§6），所以是一次性處理** |
| **G3** | `src/facecore/research/cli.py:874` header 一致性驗證 | 期待 38 欄 tuple | tuple 改 39 欄即自動一致 | **不需改程式**，隨 tuple 更新 |

⚠️ **G1 的改寫是本 spec 唯一動到「既有 append-only 保證強度」的地方。**

⚠️ **⚠️ 用「相對順序不變」是錯的 —— 那是第二輪 review 被駁回的版本，本節已修正。**

⚠️ **守護的對象是「位置」，不只是順序。** `#141` 註解寫的是「operator 的試算表公式依賴**既有位置**」。

⚠️ **反例**：若未來有人在 index 20 插入一欄，W1 三欄變成 36/37/38 —— **相對順序完全不變，但位置變了。** 而「順序不變」的斷言**會通過**，⚠️ **守護目標（位置）沒被守住。**

**G1 的最終斷言（逐字，不得弱化）：**

```python
assert G3_DEMO_RESULTS_CSV_COLUMNS[-4:] == (
    *EXPECTED_NEW,
    "operator_verdict",
), (
    "W1 must append, never insert: operator's spreadsheet formulas "
    "depend on the existing positions"
)
```

⚠️ **這比原版還強一格**：原版 `[-3:]` 只保證三欄占末三；新版 `[-4:]` **同時保證三欄仍在 35/36/37**（因為它們在長度 39 的 tuple 裡若占 `[-4:]` 的前三格，即 index 35/36/37）**且** `operator_verdict` 在最末。

⚠️ **實測現況（AST 解析，#158 追加 `operator_verdict` 之後）**：`len = 39`、`expected_count` → 35、`loaded_count` → 36、`gallery_rejected` → 37、`operator_verdict` → 38（末欄）。⚠️ **⚠️ 追加前的實測值是 `len = 38`** —— ⚠️ **本節其餘斷言（`[-4:]` 仍涵蓋 W1 三欄）是以追加後的 39 為前提。**

⚠️ **⚠️ 本 spec 之前就存在的保證邊界（非本 spec 造成）**：`test_the_23_legacy_columns_keep_their_exact_names_and_order`（`:75-79`）只 pin 住 `[:23]`。⚠️ **index 23–34 那 12 欄目前沒有任何位置守護。** ⚠️ 本 spec 不修（那是獨立的缺口），但**實作者不得順手擴張 §5.1.1 的範圍去動它**。

### 5.2 ⚠️ `label_kind` 三值契約完全不動

operator 2026-09-24 明令（`:1653-1665` 實作處）：`label_kind` 維持 `enrolled`／`unenrolled`／`uncertain` 三值，**那是 spec 契約**。

⚠️ **「正確／不正確」不得塞進 `label_kind`。** `_press_key(correct=)`（`qt_window.py:1640`）目前把 `correct` 映射成 `label_kind`：

```python
if correct:
    identity = self.desktop.display_identity()
    if identity is None:
        self.desktop.label_terminal(None, kind="unenrolled")
    else:
        self.desktop.label_terminal(identity, kind="enrolled")
else:
    self.desktop.label_terminal(None, kind="uncertain")
```

**這個映射維持不變** —— `operator_verdict` 是**獨立的第五個維度**，不重定義 `label_kind`。

### 5.3 `probe_kind` 空值與 `operator_verdict` 值域的組合必須可區分

⚠️ **⚠️ 本節在與 `runbook`（#156）比對後改寫。** ⚠️ **初版是「三種留空語意」，而現在 `presenting_identity` 不再留空**（§2.3）—— **只有 `probe_kind` 會空**，所以從三種簡化成兩種。

⚠️ **⚠️ 那一個簡化是 §2.3 造成的，不是終點** —— ⚠️ **本節後來又因另一個理由再變一次，與 §2.3 無關**：⚠️ **「刻意略過標註」原本被當成空值的一種，⚠️ 而那個設計無法與漏寫／忘按區分（三者 CSV tuple 位元組相同），⚠️ 所以現在改用明確值 `operator_verdict=skipped`，⚠️ 「刻意略過」不再是空值。** ⚠️ **⚠️ 所以本節現在講的不是「空值語意有幾種」，而是「`operator_verdict` 的值域有幾種」** —— ⚠️ **讀本節時不要回到「兩種空值」的舊讀法，詳見下表。**

⚠️ **`presenting_identity` 在 `probe_kind=nontarget` 時一律有值**（`outsider` 或清單檔名，§2.3）。⚠️ **所以「`probe_kind=nontarget` 但身分空」在實作後不可能出現** —— ⚠️ **若有，那是不合法狀態，不是誠實狀態。**

⚠️ **⚠️ `probe_kind` 空值的兩種語意，加上 `operator_verdict` 的值域，構成下表的四列** —— ⚠️ **⚠️ 表格列數不再等於「空值語意」的種類數，⚠️ 那兩件事在不同軸上。**

| `probe_kind` | `presenting_identity` | `operator_verdict` | 語意 |
| --- | --- | --- | --- |
| `target`／`nontarget` | 有值（合法值） | `correct`／`incorrect` | **已標註**，可進統計 |
| ⚠️ **空** | **空** | **空** | ⚠️ **該輪未進入標註流程** —— §4.6 的 `cancelled`／`error`，或 operator 中途中止 ⚠️ **（⚠️⚠️ **P3 必須建立的機制，現行 source 沒有** —— ⚠️ 現行 `cancelled`／`error` 輪無任何寫入路徑；⚠️ **若 P3 不建立，這一列在 CSV 中永遠對不上任何資料**）** |
| ⚠️ **待 P3 裁決** | ⚠️ **待 P3 裁決** | ⚠️ **待 P3 裁決** | ⚠️ **`invalid_input` 輪** —— ⚠️ **⚠️ 這一列的 `probe_kind` 與 `presenting_identity` 怎麼處理，本 spec 未定義，⚠️ 不得假設**（見下） |
| ⚠️ **待 P3 裁決** | ⚠️ **待 P3 裁決** | `skipped` | ⚠️ **operator 刻意略過標註** —— §4.2 的「略過」動作；⚠️ **它是明確值而非空值** |

⚠️ **⚠️ 表格共四列；⚠️ `skipped` 列與 `invalid_input` 列各有不同欄位刻意留白，本 spec 都不定義。** ⚠️ **⚠️ 兩列一律以內容指稱，不以位置指稱 —— 表格改列序時這些指稱不會失效。**

⚠️ **⚠️ `skipped` 列（`operator_verdict=skipped` 那一列）**：`probe_kind` 與 `presenting_identity` 刻意留白 —— 「略過」之後這兩欄寫推導值還是留空，是實作決定，此處不假定。⚠️ **所以 `skipped` 列不宣稱它們有值，也不宣稱它們為空。** ⚠️ **唯一已定義的是 `operator_verdict=skipped`** —— ⚠️ **那正是本裁決要表達的事實：operator 刻意不判，與漏寫、忘按不同。**

⚠️ **⚠️ `invalid_input` 列（`probe_kind`／`presenting_identity`／`operator_verdict` 三欄全留白那一列）：三欄全留白，本 spec 不定義其結果** ⚠️ **`probe_kind` 在那一輪怎麼處理（隨 status 清空／保留上一輪值／明確寫空）是 P3 的設計裁量，此處不假定、不預填。**

⚠️ **⚠️ 但 `invalid_input` 列的存在本身必須被記錄**：⚠️ **§4.3 明文規定 `invalid_input` 不問是誰（`probe_kind` 空），⚠️ 而該輪必然進標註流程、operator 按 ✓／✗ 都會寫 verdict，⚠️ 所以「`probe_kind` 空 ＋ verdict 非空」這個組合是必然可達的，⚠️ 而本表原本只有「有值 ＋ verdict」與「全空」兩列 —— ⚠️ 漏掉它等於宣稱它不可達。**

⚠️ **⚠️ 兩列的留白會互相牽動**：⚠️ **`invalid_input` 列的裁決決定了 `invalid_input` 輪的 `probe_kind` 怎麼來，⚠️ 而那正是 §4.2 判定「`invalid_input` 不提供略過」時所依賴的前提 —— ⚠️ 若 `probe_kind` 不再是空的，⚠️ 該判定的前提就需要重驗。** ⚠️ **所以 P3 裁決 `invalid_input` 列時必須連帶回頭檢查 §4.2 的可用範圍。**

⚠️ **⚠️ 下表列出三種已定義情形；⚠️ 未裁決的 `invalid_input` 情形故不在表內，⚠️ 但它的存在不可忽略（見上方 `invalid_input` 列說明）**：

| 情形 | `probe_kind` | `operator_verdict` | 下游如何處置 |
| --- | --- | --- | --- |
| 未進入標註流程 | 空 | 空 | ⚠️ **⚠️ P3 必須建立的機制，⚠️ 現行 source 不存在這種列** —— ⚠️ 若 P3 未建立，**這裡沒有任何資料可丟**，⚠️ 而非「不得丟棄一個存在的列」（§4.6） |
| 刻意略過 | ⚠️ **待 P3 裁決** | `skipped` | ⚠️ **明確的 operator 選擇，不進正確率統計** |
| 已標註 | 有值 | `correct`／`incorrect` | 可進統計 |

⚠️ **⚠️ 「漏寫／忘按」與「刻意略過」的碰撞已被 `skipped` 消除 —— 這是本節 `skipped` 列存在的理由**：⚠️ **在舊規則下，刻意略過、漏寫、忘按三者的 CSV tuple 位元組相同**（⚠️ **都寫「ground truth 有值 ＋ verdict 空」**，⚠️ **沒有任何欄位能區分**），⚠️ **所以那時無法斷言「空 verdict 就是刻意略過」** —— ⚠️ **而下游若照那個假設設計，⚠️ 就會把真正的漏寫與忘按一併讀成「operator 刻意略過」。**

⚠️ **新規則消除了這個碰撞**：⚠️ **刻意略過寫 `skipped`，⚠️ 而漏寫與忘按仍會留下「ground truth 有值 ＋ verdict 空」這個 tuple。** ⚠️ **所以現在後者可以被偵測出來** —— ⚠️ **⚠️ 而它已不再是合法狀態** ⚠️ **（§5.1）：任何進入標註流程的輪次都會寫三個值之一，未進入的輪次 `probe_kind` 又是空的。** ⚠️ **所以「ground truth 有值 ＋ verdict 空」是資料異常，⚠️ 應被驗證工具標記出來，而不是當成第三種狀態接受。**

⚠️ **「未標註」在實作後只有一個來源**：§4.6 的 `cancelled`／`error`（不顯示標註按鈕）。⚠️ **`matched`／`unknown`／`review`／`timeout`／`invalid_input` 五個值一律會進入標註流程**（§4.6 排除的是 `cancelled`／`error`），⚠️ **所以進入標註流程的輪次必寫 `correct`／`incorrect`／`skipped` 三者之一 —— 空 verdict 在那裡不可產生，operator 漏按的「髒列」在資料模型上不可產生。**

⚠️ **⚠️ 「略過」有兩個入口，兩者都必須寫 `skipped`** ⚠️ **§4.2 標註區的那顆「略過」按鈕是其中一個入口；⚠️ 另一個是標註區之外的「再次辨識」路徑（`qt_window.py` 的 `_record_unlabeled_round`）—— ⚠️ operator 看到一輪結果、選擇不判、往下走，那與 `skipped` 的定義是同一件事，⚠️ 差別只是它今天叫「再次辨識」。**

⚠️ **⚠️ 所以該路徑在新規則下應產出 `operator_verdict=skipped`** —— ⚠️ **⚠️ 這是資料模型的結論，不是要求改「再次辨識」按鈕的行為或位置** ⚠️ **該按鈕**不在** §4.2 的 ✓／✗／略過 那列，⚠️ §4.2 講的同列理由（§4.2）不涵蓋它，⚠️ 而它今天已經具備「不判定就往下走」的語意。** ⚠️ **要改的只是它落盤時寫進 `operator_verdict` 的值。**

⚠️ **⚠️ 這一條是 §5.1「不可達」斷言的必要前提** ⚠️ **現行實作在該路徑傳 `probe_kind` 與 `presenting_identity`，⚠️ 但**不傳** `operator_verdict`（預設空字串，`research/cli.py` 的 `append_g3_demo_results_csv`），⚠️ 因此它今天產生的正是「ground truth 有值 ＋ verdict 空」這個 tuple** —— ⚠️ **⚠️ 而那正是 §5.1「不可達」斷言宣告為資料異常的形狀。** ⚠️ **⚠️ 若該路徑不寫 `skipped`，⚠️ 上面的斷言就是錯的，⚠️ 而下游會開始把「再次辨識」的輪次誤標成漏寫／忘按。**

⚠️ **⚠️ 這個結論成立的前提是「每一輪都寫 `operator_verdict`」** —— ⚠️ **`skipped` 讓它成立，因為 operator 不判定時也有一個明確值可寫；⚠️ 在舊規則下同一個結論會被「刻意略過寫空」這件事推翻。** ⚠️ **⚠️ 但「`operator_verdict` 全表不可能空」仍不成立** —— ⚠️ **§4.6 的 `cancelled`／`error` 輪次兩欄都空，⚠️ 而那是合法的執行結果紀錄，⚠️ 必須保留（見下一段）。**

⚠️ **⚠️ 為什麼這個區分對下游工具是關鍵（第二輪 review 指出）**：⚠️ 若分析工具以為「`probe_kind` 空 = operator 漏按」去設計，它會等一種**永遠不會出現**的列，⚠️ **而真正的 `cancelled`／`error` 列會被它誤判成「不該出現的髒資料」而丟棄** —— ⚠️ **那會靜默吞掉中斷與錯誤的輪次，而那正是診斷 run 最需要保留的失敗紀錄。**

⚠️ **區分方式就是這兩個欄位的組合**（空值必然成對出現）。⚠️ **下游分析工具必須依此組合過濾**：不得把「空」當成 `nontarget`，⚠️ **也不得把 `cancelled`／`error` 當成髒資料丟棄** —— ⚠️ **⚠️⚠️ 而 `cancelled`／`error` 那一列是 P3 必須建立的機制，⚠️ 現行 source 沒有寫入路徑（§4.6），⚠️ 所以「不要丟棄」的對象在今天的 CSV 裡不存在，⚠️ 但 P3 建立後必須保留它。**

⚠️ **⚠️ 本 spec 落地時需同步更新 「可達組合」表（`### 現在的狀態是分側的` 子節內）** —— ⚠️ **該表現在列了四種組合**（含「`probe_kind=nontarget` ＋身分空」「漏填」兩列），⚠️ **本 spec 實作後那兩列都不再可能出現**。⚠️ **那是實作 PR 的連帶範圍，不是本 spec 的落地範圍。**

---

## 6. 與既有守護測試的關係

| 守護 | 處置 |
| --- | --- |
| `tests/live/test_d7_w0b_writer_wiring.py`（3 條，`#154`） | ⚠️ **保留全部，且不需改測試** —— 處置見 §6.2（**這是 plan 階段才發現的機制，本節初版誤述**）。 |
| `tests/live/test_d7_w0b_probe_kind_input.py`（`#154`，377 行） | ⚠️ **重寫，不保留原七支測試；五個 helper 原地重用。** 見 §6.1。 |
| `tests/cli/test_d7_w1_gallery_visibility.py:69-73`（末三欄） | ⚠️ **改寫斷言**，保留 append-only 意圖。見 §5.1.1 G1。 |
| `tests/cli/test_d7_w1_gallery_visibility.py:65-67`、`:75-79` | ✅ **不需改** —— 存在性斷言與前 23 欄斷言皆不受追加影響。 |
| `tests/live/test_g3_commit_on_label.py` | 需檢查是否依賴 `label_kind` 映射（應不受影響） |
| `cli.py:874` header 一致性驗證 | ✅ **不需改程式**，隨 tuple 更新自動一致（§5.1.1 G3）。⚠️ **spec 初版誤列為「必須更新」，已修正。** |

⚠️ **`test_d7_w0b_probe_kind_input.py` 已有 377 行且是 qt-smoke job 的顯式檔案清單成員**（`.github/workflows/ci.yml:171`）—— 重寫時**不得**順手把它改成新檔（那需要改 `ci.yml`，且是本專案反覆踩到的「守護存在但沒有任何 CI job 執行它」的成因）。

### 6.2 ⚠️ `writer_wiring` 為何零改動即可保留（plan 階段才確認的機制）

⚠️ **本節是寫執行計畫時才發現的 —— spec §6 初版誤述「守護的 AST 結構斷斷言仍成立」，那句太籠統，⚠️ 實際機制如下。**

**三條守護各自的比對對象：**

| 守護 | 比對什麼 | 移除下拉後 |
| --- | --- | --- |
| `test_both_demo_row_paths_are_present` | **writer 所在的函式名**（`_press_key` / `_record_unlabeled_round`，`_EXPECTED_WRITERS`） | ⚠️ **需追蹤 `_record_unlabeled_round` 的 `operator_verdict` 語意** —— ⚠️ 該 writer 今天不傳 `operator_verdict`，⚠️ 而本 spec 規定它應產出 `skipped`（§5.3），⚠️ **⚠️ 「不受影響」的前提是 verdict 語意不變，⚠️ 而本次變更恰恰改了 verdict 語意。** |
| `test_every_writer_forwards_both_ground_truth_columns` | **kwarg 名稱**（`probe_kind` / `presenting_identity`） | ✅ 不受影響 |
| `test_no_writer_passes_a_constant_ground_truth` | ⚠️ **accessor 名稱**（`_ACCESSORS = frozenset({"_probe_kind_value", "_presenting_identity_value"})`，比對方式 `return _callee(value) in _ACCESSORS`） | ⚠️ **名字必須保留** |

⚠️ **第三條是**名字比對**，不是行為檢查** —— ⚠️ 若實作時順手把 accessor 改名（例如改成 `_operator_verdict_value`），**哪怕行為完全正確，這條會轉紅。**

**裁決：保留兩個 accessor 的名字（`_probe_kind_value` / `_presenting_identity_value`），只改其內部實作**（從按鈕／已決定值讀，而非從 combo 讀）。

⚠️ **理由**：這條守護保護的**意圖**是「值必須在寫入那一刻被讀出來」（其 docstring：`A forwarding site must call these rather than pass a constant: the columns are per-round, and 「per round」 is enforced only by reading them at write time`）—— ⚠️ **那個性質在 accessor 內部改讀按鈕狀態之後完全成立**，⚠️ **因為它仍然是 write-time read。** ⚠️ **零測試改動，且不犧牲任何保護。**

⚠️ **⚠️ 命名誤導的取捨**：`@property _probe_kind_value` 讀的不是 combo 了。⚠️ **接受的代價** —— ⚠️ **但實作者必須在 accessor 的 docstring 裡寫明「值來自 operator 的按鈕標註，不是 combo 選取」**，⚠️ **否則未來讀者會以為它讀 combo 而誤判。**

⚠️ **⚠️ 本節的寫法教訓**：⚠️ **「守護會不會紅」不能用「守護斷言的大意看起來仍然成立」來回答** —— ⚠️ **必須讀守護的比對機制（比對名字還比對行為）。** ⚠️ **spec §6 初版就是犯這個錯，差點讓 plan 建在一個無法達成的驗收條件上（§7.2 第 8 條）。**

### 6.1 `test_d7_w0b_probe_kind_input.py` 重寫範圍（採納 lead 建議）

⚠️ **五個 helper 全部保留、原地重用** —— `_has_qt()` skipif（`:39-50`）、`_gallery()`／`_profile()`／`_desktop()`／`_window()`／`_round_row()`（`:59-179`）。

⚠️ **它們是真實 wiring**（`FakeCapture`、真 `DesktopSession`、真 `_QtResearchWindow`），⚠️ **換掉等於重建，這是該檔最值錢的部分。**

⚠️ **七支測試全部作廢** —— 它們測的是「兩個每輪下拉可獨立設定」，新 UI 沒有下拉。

⚠️ **`:283` 那支自稱 "End-to-end" 但直接呼叫 `g3_demo_round_row`、不經 Qt** —— ⚠️ **那個 docstring 本身就是過度宣稱**，新測試不得沿用。

**新增 `_press_key` 的端到端路徑測試** —— ⚠️ **這是 spec review 後才確定的**：原本獨立存在的 ④（`_press_key` behavioral 覆蓋）**併進本實作**。

⚠️ **理由**：④ 的介面是「測按正確／錯誤 → CSV」，⚠️ **在新 UI 形狀定案前不該動**，⚠️ **而現在 spec 已定義 UI，這是唯一安全的時機。**

⚠️ **`_press_key` 現況已實作按鈕到 `label_kind` 的映射**（`qt_window.py:1640-1665`，且 fail-closed：`標註未綁定，無法落盤`），⚠️ **新測試要走這條真實路徑，不直接呼叫 `g3_demo_round_row`** —— 否則重蹈 `:283` 的過度宣稱。

---

## 7. Acceptance

### 7.1 功能

1. 五格模型（§2）每一格的 `probe_kind`／`presenting_identity`／`operator_verdict` 都符合表中值。
2. 「結果正確」路徑**零額外輸入**即可推進下一輪。
3. 「結果錯誤」路徑的輸入區**只在按了錯誤之後**出現。
4. non-target 清單由**程式讀資料夾**產生；**換掉資料夾內容不改任何程式碼**即可生效。
5. `invalid_input` 輪只出現 ✓／✗，不出現身分輸入。
6. ⚠️ **「operator 刻意略過標註」由明確值 `operator_verdict=skipped` 記錄，不與空值混用**（§4.2、§5.1）。
   ⚠️ **空 verdict 只代表「該輪未進入標註流程」（§4.6 的 `cancelled`／`error`，或中途中止），⚠️ 而該情形可由 `probe_kind` 為空辨認。** ⚠️ **⚠️⚠️ 而這一條描述的是 P3 必須建立的機制，⚠️ 現行 source 的 `cancelled`／`error` 輪沒有任何寫入路徑** —— ⚠️ **若 P3 不建立，⚠️ 這個「可辨認」在 CSV 中永遠對不上資料。**
   ⚠️ **（本條初版寫「兩種空值語意」，⚠️ 那個設計把「刻意略過」當成空值的一種，⚠️ 而空值無法與漏寫／忘按區分。⚠️ 本 spec 後來改採明確值 `skipped`，⚠️ 該混淆已消除 —— ⚠️ 而 §2.3 只處理 `presenting_identity` 不留空，⚠️ 與 `operator_verdict` 的演進無關，⚠️ 不要把本條的修正歸給它。）**

### 7.2 不破壞

7. `label_kind` 三值契約不變（`_press_key` 內的映射逐字不動）。
8. ⚠️ **`writer_wiring` 三條守護全部保留且通過，且測試檔零改動** —— ⚠️ **機制見 §6.2：第三條是「accessor 名字比對」，所以兩個 accessor 名字必須保留**（`_probe_kind_value` / `_presenting_identity_value`），只改內部實作。
9. `test_each_round_can_set_its_own_value` 的時間命題仍被守護（**設定方式改為按鈕，命題本身不變**）。
10. ⚠️ **CSV 欄位順序：既有 38 欄的相對順序與索引不變，新欄在最後。**
    ⚠️ **⚠️ 這一條與三條既有守護衝突，處置見 §5.1.1** —— 追加必然使 G1（末三欄斷言）、G2（`widths == {38}`）轉紅，⚠️ **兩者改寫是「追加」的必然後果，不是本 spec 放棄 append-only。**
11. `test_d7_w0b_probe_kind_input.py` 仍在 `ci.yml` 的 qt-smoke 清單內被執行（**重寫不得換檔**）。
12. ⚠️ **證明測試被 CI 執行時，必須用該測試實際會跑的 job**（計數或 collect-only 實測，不是只讀 workflow 檔）。
    ⚠️ **（plan 階段實測：該檔在 `verify` job 是 `5 skipped`、在 `qt-smoke` job 是 `5 passed` —— ⚠️ 用 `verify` 的數字證明它是壞證據。）**
13. ⚠️ **`G1` 的改寫必須保留「W1 三欄位置」的斷言**（`[-4:]` 逐字版）—— ⚠️ **「相對順序不變」已被駁回**（那守不住位置，§5.1.1）。
14. ⚠️ **`cancelled`／`error` 輪不出現標註按鈕**（§4.6）。

### 7.3 禁止事項（operator 既有約束）

13. **不得**在任何地方硬編碼 `nontarget-01`…`nontarget-30` 或任何檔名。
14. **不得**用自由文字取代下拉（§4.3）。
15. **不得**用 `display_identity()` 或任何系統判斷自動填 `presenting_identity`（§1.1 的循環理由）。
16. **不得**動門檻值（`match_threshold 0.363`／`margin 0.1`／`required_support 3`）、品質門、三幀規則、support 視窗。
17. **不得**動 `label_kind` 三值。
18. **不得**把 non-target 檔名、per-batch 數字寫進 repo。
19. **不得**把 `probe_kind`／`presenting_identity` 插在 38 欄的中間位置。
20. **不得**對 CSV 既有欄位改名或刪除。

---

## 8. 未解與停止條件

### 8.1 本 spec 不解決

- ⚠️ **格子 3 的歸因維度**（哪一個 non-target 最容易被誤認）—— §2.1 已載明失去。需要對照表才拿得回來。
- ⚠️ **無效輪的漏檢率**有記錄（`operator_verdict=incorrect`），但**漏了誰**沒有 —— §3.1 已載明。
- ⚠️ **⚠️ `cancelled`／`error` 輪的未解問題（P3 必須回報裁決，不得自行假設）** —— ⚠️ **現行 source 的這一輪在 CSV 裡沒有任何痕跡**（實測見 §4.6），⚠️ 因此：
  - ⚠️ **⚠️ operator 要怎麼知道「這一輪被取消了」—— 尚未裁決。** ⚠️ 是要在 UI 上提示、還是靜默讓下一輪取代、還是別的機制，⚠️ **本 spec 不決定。** ⚠️ **⚠️ 若不處理，operator 可能以為自己只是漏按了標註按鈕，⚠️ 而實際是那一輪被系統取消。**
  - ⚠️ **`cancelled`／`error` 輪應不應該被記錄 —— 尚未裁決。** ⚠️ §4.6 與 §5.3 那張表描述的是**目標狀態**，⚠️ 而現行 source 沒有寫入路徑；⚠️ **要寫一列記著失敗（§4.6 的原意），還是完全不寫（與停止相機同）—— 兩者都未被裁決。** ⚠️ **⚠️ 而這個決定會連帶影響 §5.3 表格是否需要那一列。**
  - ⚠️ **形狀與 `invalid_input` 那一列相同**（§5.3 第三列）：⚠️ **同樣是「存在這個組合但結果未定」，⚠️ 同樣不得由實作者預填。**

### 8.2 需 operator 在場

- ⚠️ **W0-b 真機跑**（30 張 non-target 逐張）：需 operator 在場操作。⚠️ 本 spec 完成後，流程手冊（`~/Downloads/W0-b-跨身分實驗執行手冊.md`）**需要更新** —— 那份手冊從未提及這兩欄（實測 `probe_kind`／`presenting_identity`／「留空」全部零命中）。⚠️ **那是 operator 的私人文件，需要他授權才能寫。**

### 8.3 停止條件

⚠️ **若實作過程發現存在「不依賴 `presenting_identity`」的 `probe_kind` 推導路徑** —— ⚠️ **停止並回報**，因為那會改變 §2 的模型（目前認定兩欄是一個事實的兩面，無獨立推導路徑）。

⚠️ **（spec review 後收緊）**：原版寫「若發現程式能自動推導 `probe_kind` 就停」是**死條件** —— ⚠️ 按五格模型，`probe_kind` 的可推導性天生依附在 `presenting_identity` 上（要知道這輪是誰才能判他是否註冊），所以那個條件在現況下幾乎不可能觸發。

⚠️ **結構性排除的依據**：`qt_window.py:571` 的判斷式 `probe_kind == "nontarget"` 是**精確字串比對**，⚠️ 那意味著 `probe_kind` 的值域就是封閉的兩個字串 ＋ 空，**沒有任何第三條自動推導路徑的入口**。

### 8.4 已在 spec review 中裁決的事項

⚠️ **記錄本 spec 的裁決歷程，避免未來重開已決問題：**

| 事項 | 裁決 | 依據 |
| --- | --- | --- |
| §1.1 循環論證是否成立 | ✅ **成立，spec 前提不變** | lead 2026-10-05 獨立判斷認同（「假 ground truth 比空欄位危險」） |
| 五格是否完整 | ✅ **完整，不需第六格** | `session.py:852-859` 的 `face_count == 1` 硬前提 |
| G1 append-only 守護要不要鬆綁 | ✅ **鬆綁斷言、保留意圖**（§5.1.1） | 追加是既定方向；鬆綁的是斷言形式不是保證強度 |
| `probe_kind` 空值涵蓋哪些值 | ✅ **`unknown`／`review`／`timeout` 三值全含**（§4.3） | operator 看到三者都是「沒找到此人」（都不顯示名字） |
| `probe_kind_input.py` 重寫範圍 | ✅ **五 helper 留、七測試作廢、併入 ④**（§6.1） | lead 建議，理由採納 |
| G1 斷言強度 | ✅ **`[-4:]` 逐字版**（§5.1.1） | 「相對順序不變」不等於「位置不變」——lead 駁回第二版 |
| §5.3 `skipped` 列與 `invalid_input` 列的定義 | ✅ **「該輪未進入標註流程」或「operator 刻意略過（`skipped`）」**（§5.3） | ⚠️ **新 UI 下空 verdict 只在 `probe_kind` 空時可能** —— ⚠️ **`skipped` 是明確值而非空值，⚠️ ground truth 有值而 verdict 空是資料異常，「漏按」不再是死列而是可被偵測的例外** |
| ⚠️ **`nontarget` 側值域** | ✅ **決定為 `outsider`，不留空**（§2.3） | ⚠️ **落地時與 `runbook`（#156）交叉比對才發現的衝突** —— runbook 該列寫「不可留空」，spec 初版設計成留空 |

---

## 附錄 A：`runbook` 連帶修改指示（落地 PR 必做，**僅此三處**）

## ⚠️⚠️ **錨定方式：內容錨定為準，行號僅供當次參考**

⚠️ **本附錄初版以行號錨定，那是不安全的 —— 已修正。**

⚠️ **行號會位移**：本 spec 寫成時的行號是 `:541`／`:543`／`:521-534`，⚠️ **而 `#156`（`12f0bd6`）merge 之後實際是 `:545`／`:547`／`:530-535`** —— 已由 lead 在 worktree 實測確認（commander 獨立複驗）。

⚠️ **⚠️ 這不是筆誤，是結構性問題**：**squash merge 會讓行號位移，而這份 spec 會長期在 repo 裡。** ⚠️ **任何以行號為準的指示，都會在未來某次 merge 後指向錯誤的位置。**

⚠️ **所以：本附錄每一處用「節標題 ＋ 表頭／欄名」錨定，行號只寫在括號裡當次參考。** ⚠️ **實作者必須先用內容確認位置，再動手。**

⚠️ **⚠️ 特別警告：不得只依行號範圍操作。** ⚠️ 初版的「不動 `runbook:521-534`」若被照字面執行，⚠️ **會落在節標題「### 現在的狀態是分側的」（`:521`）與表格本體（`:530-535`）之間** —— ⚠️ **而那正是 `#156` 剛 merge、措辭已經過 review 的段落。**

### A.0 三處的定位方式（**先讀這裡**）

| 附錄 | 內容錨定（**以此為準**） | 行號（`12f0bd6`，僅供參考） |
| --- | --- | --- |
| **A.1** | 在標題為 **`## \`probe_kind\` / \`presenting_identity\`：分側狀態`** 的那一節內、**子節標題 `### W0-b 排程時的記錄規則`** 的表格中，**第一欄為 `presenting_identity`** 的那一列 | `:545` |
| **A.2** | 同上那張表，**第一欄為 `填不進去怎麼辦`** 的那一列 | `:547` |
| **A.3** | 在子節標題 **`### 現在的狀態是分側的`** 之下，**表頭第一欄為 `可達組合`** 的那張表（表頭行 ＋ 分隔行 ＋ 四列本體） | `:530-535` |

⚠️ **A.1 與 A.2 是同一張表的相鄰兩列** —— ⚠️ **⚠️ 實作者必須確認兩列同時改對**，⚠️ **因為它們是同一個語意塊（值域已定義 → 所以 nontarget 側可填）。**

⚠️ **⚠️ 那張表下方有一個 `> ⚠️ **本節不再引用 \`cli.py\` 的註解作為權威。**` 的 blockquote（`#156` 新增）** —— ⚠️ **⚠️ 不要動它**，⚠️ **它是該節的一部分，且已經過 review。**

### A.1 `presenting_identity` 那列（`:545`）

**現況**：

> `presenting_identity`｜ 測試者在註冊組裡就填其身分（例如 `enroll-07`）；**沒有註冊的測試者，這欄填什麼必須在 W0-b 排程時定義**（例如 `outsider` 之類的標記），不可留空

**改為**：

> `presenting_identity`｜ 測試者在註冊組裡就填其身分（例如 `enroll-07`）；**沒有註冊的測試者填 `outsider`** —— 這個值已於 **2026-10-05** 決定（見 `docs/specs/2026-10-05-g3-operator-verdict-ground-truth.md` §2.3）。⚠️ **若 operator 手邊有 non-target 對照表，改填該張照片的檔名 stem**（App 會讀 non-target 資料夾列出候選）。**不可留空。**

⚠️ **理由**：該列原本把值域定義推給「W0-b 排程」，⚠️ **而 spec §2.3 就是那個決定**，所以這一列的狀態要從「待定義」改成「已定義」。

### A.2 `填不進去怎麼辦` 那列（`:547`）

**現況**：

> 填不進去怎麼辦｜ **分側看。** `target` 側的下拉就是填入路徑，#154 之後可直接選。`nontarget` 側的 `presenting_identity` **目前仍填不進去**，因為它的值域還沒定義 —— 那是 W0-b 排程要決定的，不是 W0-a 的缺口，也不要在 App 裡繞過它

**改為**：

> 填不進去怎麼辦｜ ⚠️ **兩側都已可填（spec 實作後）。** `target` 側是註冊組下拉；`nontarget` 側是 `outsider` 或 non-target 清單。⚠️ **若 App 尚未實作該 spec，`nontarget` 側仍會是 disabled 的「尚未定義」項目 —— 那是預期狀態，不要在 App 裡繞過它。**

⚠️ **理由**：該列說「那是 W0-b 排程要決定的，不是 W0-a 的缺口」⚠️ **現在已決定**，但 ⚠️ **spec 落地 ≠ 程式已改**，⚠️ **所以要同時寫明「實作後」與「實作前」的兩種狀態**，⚠️ **否則 operator 在程式改完前會讀到過期描述而在 App 裡亂繞。**

### A.3 ⚠️ **「可達組合」表（`:530-535`）—— ⚠️ 落地 PR 不動，留到實作 PR**

⚠️ **定位**：子節標題 `### 現在的狀態是分側的` 之下，表頭第一欄為 `可達組合` 的那張表。

⚠️ **該表现在列四種組合**，其中兩種在 spec 實作後不再可能出現：

| 現況列 | 實作後 |
| --- | --- |
| `probe_kind=nontarget` ＋ 身分空（值域未定義） | ❌ **不再可能**（值域已定義為 `outsider`） |
| `probe_kind` 空 ＋ **有值**（「漏填」） | ⚠️ **不再可能**（新 UI 下身分由選項決定，不存在只選身分不選種類） |
| `probe_kind` 空 ＋ 空（未記錄） | ✅ 仍可能，但**語意收斂為「該輪未進入標註流程」**（§5.3） ⚠️ **⚠️ 而該機制現行 source 沒有，⚠️ `cancelled`／`error` 輪今天不寫入任何 row；⚠️ 若 P3 不建立，這一列在 CSV 中永遠對不上資料（§4.6）** |
| `probe_kind=target` ＋ 空（「漏填」） | ⚠️ **新 UI 下不會產生**（`target` 側選項來自已載入身分，無理由空） |

⚠️ **⚠️ 為什麼落地 PR 不動這張表**：⚠️ **spec 落地時程式還沒改**，⚠️ **那四種組合當下仍然全部存在。** ⚠️ **若落地 PR 就把它們標成「不可能」，`runbook` 會描述一個尚未發生的狀態** —— ⚠️ **那正是這幾輪反覆修的「文件領先於程式」那型病。**

⚠️ **所以這張表的更新是實作 PR 的連帶範圍。**