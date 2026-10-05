# G3-w：operator 結果標註與 ground truth 自動帶出（執行計畫）

- 日期：2026-10-05
- 作者：fc-team-lead
- 建議 repo 落點：`docs/plans/2026-10-05-g3-operator-verdict-plan.md`
- 狀態：**v1 —— 兩處阻塞已由 commander 於 2026-10-05 裁決（見 §6）；階段劃分待 commander 裁決後才派 impl**
- spec 權威：`docs/specs/2026-10-05-g3-operator-verdict-ground-truth.md`（520 行，2026-10-05 三輪 spec review 後定案）
- base HEAD：`12f0bd639ed3fa31dd0014d164a14c30690b9dfd`（`#156` merge）
- ⚠️ **本檔是拋棄式執行文件**：功能權威是 spec 樹；執行完不維護、不回填。本檔不構成產品 API 承諾。

---

## 0. 本計畫的兩條硬紀律

### 0.1 ⚠️ 每個檔案錨點用「內容錨定 ＋ 行號並列」，不得只靠行號

⚠️ **本專案已因此栽過兩次**：

1. `#156` 那輪，`runbook` 的掛起句因前面段落增行從 `:535`／`:536` 位移到 `:543`／`:544`／`:545`，⚠️ **實作者第一次取數就取錯行**，事後靠 `grep -n` 重新定位才確認 byte-identical。
2. spec 附錄 A 初版寫 `runbook:541`／`:543`／`:521-534`，⚠️ **`#156` merge 後實際是 `:545`／`:547`／`:530-535`** —— 差 4 行。⚠️ 若照字面執行「不動 `:521-534`」，會落在**節標題與表格本體之間**，而那正是剛 merge、措辭已過 review 的段落。

⚠️ **squash merge 會讓行號位移**（`d549a4f9^{tree}` == `897a831^{tree}` 但 commit 不同）。⚠️ **所以本計畫所有錨點寫成「節標題 ＋ 表頭／欄名 ＋ 行號」，行號只當次參考。**

### 0.2 ⚠️ 每階段的驗收必須是「可執行命令 ＋ 預期輸出」，不是「測試通過」

⚠️ **理由**：`CLAUDE.md`（`#155` merge，`d549a4f9`）的 `## Documentation Discipline` 已立規則 —— **記 provenance，不記證據的數值內容**。⚠️ 而「測試通過」這種驗收既不可重現也不可稽核。

⚠️ **但反向也成立**：⚠️ **綠色結果在沒有對應紅色變異實驗時不構成守護有效之證據**（本專案 D0 §11 第 15 項記錄過的失效形狀）。⚠️ **所以新增／改寫的守護，該階段的驗收必須包含「把它弄壞 → 確認轉紅」。**

---

## 1. CI 現況實測（階段劃分的依據）

⚠️ **本節全部是實測值，不是讀 workflow 檔推導的。** ⚠️ 量測方式見 §1.1。

### 1.1 ⚠️ 量測方法：必須用真正隔離的 venv

⚠️ **`uv sync --extra dev` 不會移除先前安裝的套件。** ⚠️ 我第一組量測因此是壞的 —— 兩次都跑在同一個已污染的 `.venv`，得出「無差額」的假結論。

⚠️ **正確做法**（本計畫所有 CI 相關驗收都用這個）：

```bash
rm -rf /tmp/venv-verify && uv venv /tmp/venv-verify --python 3.14
VIRTUAL_ENV=/tmp/venv-verify uv pip install -e '.[dev]'
/tmp/venv-verify/bin/python -m pytest <檔> -q          # verify job 環境
/tmp/venv-verify/bin/python -c "import PySide6"        # 應 ModuleNotFoundError
```

⚠️ **CI job 的 extra 對照**（`ci.yml`）：

| job | 安裝指令 | 有 PySide6？ |
|---|---|---|
| `verify` | `pip install -e .[dev]` | ❌ |
| `qt-smoke` | `pip install -e .[dev,research-ui]` | ✅ |

### 1.2 三支受影響測試的實測結果

| 測試檔 | verify job（`.[dev]`）| qt-smoke job（`.[dev,research-ui]`）| 備註 |
|---|---|---|---|
| `tests/live/test_d7_w0b_probe_kind_input.py` | **5 skipped** | **5 passed** | ⚠️ **verify job 對它只報 skip** |
| `tests/live/test_d7_w0b_writer_wiring.py` | **3 passed** | — | PySide6-free，**verify job 會跑** |
| `tests/cli/test_d7_w1_gallery_visibility.py` | **16 passed, 1 skipped** | 在 qt-smoke 的 `Run offscreen synthetic Qt smoke` 步驟顯式清單內（`grep -n 'test_d7_w1_gallery_visibility.py' .github/workflows/ci.yml` 定位）| 兩處都跑，⚠️ **但 verify 少一支**（見下）|


⚠️ **⚠️ 為什麼 `test_d7_w1_gallery_visibility.py` 在 verify job 少一支**：

⚠️ **它沒有 module-level `skipif`，但有一支測試在函式體內 `pytest.skip`** —— `grep -n 'find_spec("PySide6")' tests/cli/test_d7_w1_gallery_visibility.py` 可定位，⚠️ 該處在 `import PySide6.QtWidgets` 前先 `pytest.skip("needs the Qt demo path")`。

⚠️ **所以 `grep -cE '^\s*def test_'` 給的 17 是「檔案裡有幾支測試」，⚠️ 不是「verify job 會跑幾支」。** ⚠️ **`--collect-only` 也看不出來** —— 函式體內的 skip 在執行期才發生，collect 階段照樣收。⚠️ **要量這件事必須實跑 `-rs` 並看 `SKIPPED` 行。**

⚠️ **兩個直接後果**：

1. ⚠️ **`test_d7_w0b_probe_kind_input.py` 的重寫，證明它被執行只能靠 qt-smoke job。** ⚠️ **用 verify job 的數字證明它是壞證據** —— 該檔在 verify 是 module-level skipif，⚠️ 只會看到 skip 數變化，看不到 pass。
2. ⚠️ **`writer_wiring` 是 PySide6-free → verify job 就跑它。** ⚠️ **所以它若因 UI 改動轉紅，verify job 立刻抓到，不必等 qt-smoke。** ⚠️ 這是 #154 當初刻意讓它 PySide6-free 的效果。

⚠️ **⚠️ 而第 1 條對 `test_d7_w1_gallery_visibility.py` 不適用** —— ⚠️ 它是函式體內 skip，⚠️ **verify job 會跑其中 16 支**，⚠️ 所以它轉紅時 verify job 看得到（⚠️ 但只會看到 16 支的結果，⚠️ 那支需 Qt demo path 的要等 qt-smoke）。

⚠️ **本表的計數何時失效、該由誰處理**：

- ⚠️ **計數本身是量測記錄，不是永久權威。** ⚠️ 它們的作用是決定 P4 要靠哪個 job 證明自己被執行，⚠️ **而那正是 D0 §11 第 15 項記錄過的失效形狀**（守護寫了但沒有任何 CI job 執行它，而 repo 仍全綠）。
- ⚠️ **`test_d7_w0b_probe_kind_input.py` 在 P4 重寫之後，本表的計數全部失效。**
- ⚠️ **⚠️ P4 必須「刪掉」本表的計數，不是「標為過期」。** ⚠️ 標為過期等於留一個假權威 —— ⚠️ 讀者看到數字加一個「已過期」標記，仍會拿那個數字當基線，⚠️ **而那正是 `PROJECT-STATE.md` 曾出現過的病（同一句既說「已刪除」又保留刪除前的數字，三輪 review 沒抓到）。**
- ⚠️ **刪掉計數不會讓本節失效**：⚠️ 本節的**命題**是「P4 的重寫只有 `qt-smoke` 會跑、`verify` 只會 skip」—— ⚠️ **那是 CI job 的結構事實，數字刪掉後仍然成立。** ⚠️ **過期的是那兩個計數，不是命題。**
- ⚠️ ⚠️ **本條不得只存在於本檔** —— ⚠️ 「P4 完成後要刪這些計數」若不寫進 P4 的 dispatch，⚠️ **會隨 session 斷裂而消失。**

### 1.3 qt-smoke 的執行模型（⚠️ 這是本專案反覆踩坑的地方）

⚠️ **`qt-smoke` 的 `Run offscreen synthetic Qt smoke` 步驟是「明確檔案清單」，從不跑 `tests/` 全套。** ⚠️ 新增測試檔若不在清單內，**不會被任何 CI job 執行**，⚠️ 而 repo 仍全綠（D0 §11 第 15 項記錄過一次，`#154` 又撞過一次）。

⚠️ **所以本計畫的硬規則**：⚠️ **任何新增／重寫的測試，必須落在既有 `ci.yml` 顯式清單內的檔案。** ⚠️ 確實需要新檔時，必須同時改 `ci.yml`，⚠️ **且該階段的驗收必須用 §1.1 的隔離 venv 證明新檔確實被 collect**（`--collect-only -q` 的數字），**不得只讀 workflow 檔。**

---

## 2. 階段劃分

⚠️ **每個階段是一個獨立 PR，可獨立 merge 且 repo 綠。**

| # | 階段 | branch 命名建議 | 類型 | 前置 |
|---|---|---|---|---|
| **P1** | spec 落地 ＋ `runbook` 連帶兩處 | `g3w-spec-landing` | 純文件 | — |
| **P2** | 資料模型：`operator_verdict` 欄 ＋ 三條守護改寫 | `g3w-operator-verdict-column` | 程式 | P1 |
| **P3** | UI：五格標註流程 ＋ non-target 清單 ＋ 移除舊下拉 | `g3w-verdict-ui` | 程式 | P2 |
| **P4** | `test_d7_w0b_probe_kind_input.py` 重寫 ＋ `_press_key` 端到端（R1） | `g3w-verdict-ui-tests` | 測試 | P3 |

⚠️ **P2 在 P3 之前的理由（比「中間會不一致」更硬）**：

⚠️ `cli.py` 裡 `grep -nF 'writer.writerow({key: row[key] for key in G3_DEMO_RESULTS_CSV_COLUMNS})'` 那行是 `writer.writerow({key: row[key] for key in G3_DEMO_RESULTS_CSV_COLUMNS})` —— ⚠️ **它只取 tuple 內的 key。** ⚠️ 所以若 UI 先做而欄位後加，⚠️ **`operator_verdict` 在 tuple 加進去之前根本不會被寫入 row，而那看起來完全正常。**

⚠️ **這是「靜默失敗」型，比「不一致」更難抓** —— 不一致會讓測試紅，靜默漏寫不會。

⚠️ **P4 在 P3 之後的理由**：⚠️ 那支測試的 5 支現況全部直接操作 `win.probe_kind_combo`／`win.presenting_identity_combo`，⚠️ **P3 移除下拉後它們必然全紅** —— ⚠️ 所以它不能與 P3 同 PR（否則 P3 的「repo 綠」無法達成），⚠️ 也不能在 P3 之前（那時新 UI 還不存在）。

---

## 3. 各階段明細

### P1 — spec 落地 ＋ `runbook` 連帶兩處

**目標**：把已定案的 spec 落進 git，並消除它與已合併 `runbook` 的兩處矛盾。

⚠️ **精確檔案清單**：

| 檔案 | 動作 |
|---|---|
| `docs/specs/2026-10-05-g3-operator-verdict-ground-truth.md` | **新增**（spec 全文，496 行）|
| `docs/plans/2026-10-05-g3-operator-verdict-plan.md` | **新增**（本檔）|
| `docs/w0a-diagnostic-run-runbook.md` | **改兩列**（見下）|

⚠️ **`runbook` 的兩處 —— 用內容錨定**（spec 附錄 A.0 的定位表）：

| 附錄 | 內容錨點 | 行號（`12f0bd6`，僅供參考）| 動作 |
|---|---|---|---|
| A.1 | 節 `### W0-b 排程時的記錄規則` 的表格中，**第一欄為 `presenting_identity`** 的那一列 | ⚠️ **不給行號** —— 定位方式：先 `grep -n '### W0-b 排程時的記錄規則' docs/w0a-diagnostic-run-runbook.md` 鎖定子節，⚠️ **⚠️ 不要直接 grep `^| \`presenting_identity\`` —— 它會命中兩行**（另一張表「### 這兩欄是幹嘛的」也有同名的第一欄），⚠️ **要取子節之後的那一行** | 改為 `outsider` 已定義的版本 |
| A.2 | 同上那張表，**第一欄為 `填不進去怎麼辦`** 的那一列 | ⚠️ **不給行號** —— 用 `grep -n '^| 填不進去怎麼辦'` 定位當前那一列 | 改為「兩側都已可填（spec 實作後）」，並寫明實作前的過渡狀態 |

⚠️ **A.1 與 A.2 是同一張表的相鄰兩列，必須確認同時改對。**

⚠️ **⚠️ 不得動的三處**：

1. ⚠️ **子節 `### 現在的狀態是分側的` 之下、表頭第一欄為 `可達組合` 的那張表**（⚠️ **不給行號** —— 用 `grep -n '^| 可達組合' docs/w0a-diagnostic-run-runbook.md` 定位表頭，向下四列即本體）—— ⚠️ **spec 落地時程式還沒改，那四種組合當下全部存在**；⚠️ 把它們標成「不可能」就是「文件領先於程式」，⚠️ **那是這幾輪反覆修的那型病。留到 P3。**
2. ⚠️ 該表下方的 `> ⚠️ **本節不再引用 \`cli.py\` 的註解作為權威。**` blockquote（`#156` 新增 —— ⚠️ **不給行號**，用 `grep -n '本節不再引用' docs/w0a-diagnostic-run-runbook.md` 定位）—— 已過 review。
3. ⚠️ `runbook` 該節的其他任何段落。

⚠️ **驗收命令**（本階段是純文件，驗收是「內容正確」而非「測試綠」）：

```bash
# 1. 範圍：只有三個檔
git diff --stat origin/main
# 預期：docs/specs/…（新增）、docs/plans/…（新增）、docs/w0a-…md（2 insertions / 2 deletions）

# 2. A.1 與 A.2 逐字等於 spec 附錄的「改為」文字
awk 'NR==545' docs/w0a-diagnostic-run-runbook.md   # 應含「沒有註冊的測試者填 `outsider`」
awk 'NR==547' docs/w0a-diagnostic-run-runbook.md   # 應含「兩側都已可填（spec 實作後）」

# 3. 可達組合表四列逐字未動
git diff origin/main -- docs/w0a-diagnostic-run-runbook.md | grep -cE '^[-+]\| .probe_kind'
# 預期：0

# 4. 三步驟（不動 Python，應無變動）
uv sync --extra dev
uv run --extra dev ruff check src tests     # 預期：All checks passed!
uv run --extra dev mypy src                 # 預期：Success: no issues found in 86 source files
uv run --extra dev pytest tests/ -q         # 預期：1212 passed, 128 skipped, 1 xfailed
```

⚠️ **review 重點**：⚠️ **逐字比對附錄 A 的「改為」文字**（commander 明寫的要求）· ⚠️ **`runbook` 那兩列直接影響 operator 在 W0-b 的操作**，⚠️ 不得因為「純文件」就降低強度。

---

### P2 — 資料模型：`operator_verdict` 欄

**目標**：在 CSV 加第 39 欄，並處理三條會被打破的既有守護。

⚠️ **精確檔案清單**：

| 檔案 | 動作 | 內容錨點 |
|---|---|---|
| `src/facecore/research/cli.py` | tuple `G3_DEMO_RESULTS_CSV_COLUMNS` **追加** `"operator_verdict"` | ⚠️ **追加為最後一欄（index 38），共 39 欄，不得插中間** |
| `tests/cli/test_d7_w1_gallery_visibility.py` | 改寫 `test_new_columns_are_appended_after_the_existing_35` | ⚠️ **斷言必須是 `[-4:]` 逐字版** |
| `tests/live/test_d7_w0b_probe_kind_input.py` | `assert widths == {38}` → `{39}` | ⚠️ 該檔在 P4 重寫，**這裡是一次性處理** |

⚠️ **⚠️ `test_d7_w1_gallery_visibility.py` 的斷言必須逐字是**：

```python
assert G3_DEMO_RESULTS_CSV_COLUMNS[-4:] == (*EXPECTED_NEW, "operator_verdict"), (...)
```

⚠️ **為什麼不能用「相對順序不變」** —— ⚠️ 那個版本**守不住位置**：⚠️ 若有人在 index 20 插入一欄，三欄變成 36/37/38，⚠️ **順序不變、斷言會通過、位置變了**。⚠️ 而 `#141`（W1）註解寫的正是「**operator 的試算表公式依賴既有位置**」。

⚠️ **`cli.py` 的 header 一致性驗證（`grep -nF 'actual_header != G3_DEMO_RESULTS_CSV_COLUMNS'`）不需改程式** —— ⚠️ tuple 改 39 欄即自動一致。

⚠️ **不得動的**：`label_kind` 三值契約（`enrolled`／`unenrolled`／`uncertain`）· 既有 38 欄的相對順序與索引。

⚠️ **驗收命令**：

```bash
# 1. 欄位數與索引（AST，不用 regex —— 註解字串會讓 regex 多算 3 欄）
uv run --extra dev python - <<'PY'
import ast
src = open("src/facecore/research/cli.py").read()
tree = ast.parse(src)
for n in ast.walk(tree):
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "G3_DEMO_RESULTS_CSV_COLUMNS":
        c = ast.literal_eval(n.value)
        assert len(c) == 39, f"應為 39 欄，實得 {len(c)}"
        assert c[38] == "operator_verdict", f"新欄應在 index 38，實得 {c[38]!r}"
        assert c[28] == "probe_kind" and c[29] == "presenting_identity", "既有欄位索引位移"
        print("OK: 39 欄，新欄在 index 38")
PY
# 預期：OK: 39 欄，新欄在 index 38

# 2. 三條守護全綠，且 writer_wiring 在 verify job 確實被執行
rm -rf /tmp/venv-verify && uv venv /tmp/venv-verify --python 3.14
VIRTUAL_ENV=/tmp/venv-verify uv pip install -e '.[dev]'
/tmp/venv-verify/bin/python -m pytest tests/cli/test_d7_w1_gallery_visibility.py -q
# 預期：16 passed, 1 skipped（⚠️ 不是 17 passed —— 見 §1.2；qt-smoke job 才是 17 passed）
/tmp/venv-verify/bin/python -m pytest tests/live/test_d7_w0b_writer_wiring.py -q
# 預期：3 passed   ← 這證明 verify job 確實執行它（PySide6-free）

# 3. ⚠️ 紅色變異實驗：把新欄從 tuple 拿掉 → G1 必須轉紅
#    （在 git archive 另開副本做，不要原地改再 cp 還原）
git archive origin/main | (mkdir -p /tmp/mut && tar -x -C /tmp/mut)
#    在 /tmp/mut 裡刪掉 tuple 最後一項，確認 test_d7_w1_gallery_visibility 轉紅
```

⚠️ **review 重點**：⚠️ **新欄是否真的在最後（AST 驗）** · ⚠️ **`[-4:]` 斷言是否逐字（不是「順序不變」）** · ⚠️ **有沒有順手動既有欄位**。

---

### P3 — UI：五格標註流程

⚠️ **⚠️ 本階段的兩處阻塞已由 commander 於 2026-10-05 裁決（見 §6）。**

**目標**：移除兩個每輪下拉，改成「✓／✗ 按鈕 ＋ 出錯時才展開輸入區」。

⚠️ **精確檔案清單**（依賴裁決結果）：

| 檔案 | 動作 |
|---|---|
| `src/facecore/live/qt_window.py` | 移除 `probe_kind_combo` ＋ `presenting_identity_combo`；新增 ✓／✗ 按鈕與展開區；`_press_key` 改為走新流程；`operator_verdict` 寫入 row |
| `src/facecore/research/cli.py` | `g3_demo_round_row` 加 `operator_verdict` 參數 |
| `docs/w0a-diagnostic-run-runbook.md` | ⚠️ **「可達組合」表（⚠️ **不給行號** —— 用 `grep -n '^| 可達組合' docs/w0a-diagnostic-run-runbook.md` 定位）在本階段更新** —— 程式改了，四種組合的存廢才成立 |
| `.github/workflows/ci.yml` | ⚠️ **僅在本階段確實需要新測試檔時改**，且必須用 §1.1 隔離 venv 證明新檔被 collect |

⚠️ **不得動的**：`label_kind` 三值契約與 `_press_key` 的既有映射（spec §5.2）· 門檻值（`match_threshold`／`margin`／`required_support`）· 品質門 · 三幀規則 · support 視窗。

⚠️ **⚠️ 兩個 accessor 必須保留原名（commander 裁決 D1-(a)）**：

| accessor | 保留名的理由 |
|---|---|
| `_probe_kind_value` | ⚠️ 它不再讀 combo，但**守護保護的性質是「write-time read」而非「讀 combo」** —— 改成讀按鈕狀態後，那個性質仍成立 |
| `_presenting_identity_value` | 同上 |

⚠️ **⚠️ 改名會讓 `writer_wiring` 的 `test_no_writer_passes_a_constant_ground_truth` 轉紅** —— ⚠️ 那條守護的 `_ACCESSORS`（在 `tests/live/test_d7_w0b_writer_wiring.py` 內 `grep -n '_ACCESSORS = frozenset'` 定位 —— ⚠️ **不要引用行號**）是 **accessor 名字的 frozenset**，`_is_live_read()`（用 `grep -n 'def _is_live_read'` 定位）回傳 `_callee(value) in _ACCESSORS` ⚠️ —— ⚠️ **它是名字比對，不是行為檢查**。

⚠️ **⚠️ 必須在兩個 accessor 的 docstring 寫明「值來自 operator 的按鈕標註，不是 combo 選取」**（commander 明寫的要求）⚠️ **否則未來讀者會以為它讀 combo 而誤判** —— ⚠️ 那個名字在 (a) 之後是有誤導性的。

⚠️ **⚠️ commit 策略：單一 commit（commander 裁決 D2）** ⚠️ **不是兩個 commit** —— ⚠️ 實測過渡狀態：先移除下拉，那 5 支直接操作 `win.probe_kind_combo` 的測試必然全紅，⚠️ **所以「每個 commit 各自讓 CI 綠」在移除下拉那個 commit 上不成立。** ⚠️ additive 骨架方案已被否決（⚠️ 那本身是 §4.1 說要移除的東西，會產生「兩套 UI 並存」的過渡 commit）。
⚠️ **PR 描述必須明確切出兩個可獨立判斷的區塊**：① 移除舊下拉與改寫 accessor ② 新增按鈕 UI 與 progressive disclosure ⚠️ **否則 reviewer 會當成一個整體改動來看。**

⚠️ **⚠️ 系統結果分支必須用實體值**（spec §4.3）：

⚠️ **`SessionStatus` 的七個值是 `matched`／`review`／`unknown`／`invalid_input`／`timeout`／`cancelled`／`error`** ⚠️ **沒有 `not_found`** ⚠️ **「沒找到此人」在程式裡是 `unknown`**（在 `src/facecore/live/session.py` 內 `grep -n 'SessionStatus.unknown'` 看那三個 return 分支），⚠️ **而標註區要覆蓋 `unknown`／`review`／`timeout` 三值** ⚠️ **`cancelled`／`error` 不進標註流程**（spec §4.4）。

⚠️ **⚠️ 靜默失敗警示**：`cli.py` 裡 `grep -nF 'writer.writerow({key: row[key] for key in G3_DEMO_RESULTS_CSV_COLUMNS})'` 那行的 `writerow({key: row[key] for key in G3_DEMO_RESULTS_CSV_COLUMNS})` 只取 tuple 內的 key —— ⚠️ **P2 已把 `operator_verdict` 加進 tuple，所以本階段只要把值放進 row 就會被寫出**；⚠️ **若忘了放，該欄會靜默留空，而 `operator_verdict` 空正是「未標註」的合法值** ⚠️ —— ⚠️ **所以驗收必須明確斷言「每輪都有值」，不能只斷言「檔案有這欄」。**

⚠️ **⚠️ 驗收命令（本階段必須包含紅色變異）**：

```bash
# 1. qt-smoke job 確實執行（那是唯一能跑 PySide6 測試的地方）
uv sync --extra dev --extra research-ui
uv run --extra dev --extra research-ui python -m pytest tests/live/test_qt_window.py -q
# 預期：全綠

# 2. ⚠️ 紅色變異：把 _press_key 的 operator_verdict 寫入移除（ast.parse 須仍通過）
#    → 必須有新測試轉紅（見 P4；P3 完成時該測試還在 P4）
#    ⚠️ P3 階段此項由 P4 補上，P3 只能靠現有測試 + 靜默失敗的人工確認

# 3. ⚠️ 靜默失敗的人工確認（不可省略）—— ⚠️ 兩欄都要，且值域要對
#    ⚠️ operator_verdict 空是「未標註」的**合法值**，所以「非空」不足以證明有寫入；
#      必須斷言值屬於 {correct, incorrect}。
#    ⚠️ probe_kind 在「按 ✓ 的零輸入路徑」上同樣可能被漏寫 —— spec §4.2 明寫
#      「正確的格子仍然要寫 probe_kind —— 程式從系統結果推導」。
#      ⚠️ **operator_verdict 非空不等於 probe_kind 有值**，兩者是獨立的靜默失敗。
#    ⚠️ 驗收方式：實際跑一輪 App，對產生的 demo CSV 逐列檢查
#      - 每列 operator_verdict ∈ {correct, incorrect}（不得為空）
#      - 每列 probe_kind ∈ {target, nontarget}（不得為空，含按 ✓ 的那一列）
#      - 若有列為空 → 這是靜默失敗，不是「未標註」
```

⚠️ **⚠️ P3 的驗收弱點（如實記錄）**：⚠️ **P3 沒有自己的新測試**，⚠️ 它靠 P4 的端到端測試來證明。⚠️ **所以 P3 與 P4 必須在同一個 PR loop 內連續驗收**，⚠️ 否則 P3 是「沒有守護的程式改動」—— ⚠️ **這是本計畫已知的弱點，不是遺漏。**

---

### P4 — 測試重寫 ＋ `_press_key` 端到端（R1）

**目標**：重寫 `test_d7_w0b_probe_kind_input.py`（377 行），並補上 R1 —— `_press_key` 真實路徑的端到端覆蓋。

⚠️ **⚠️ 不得換檔** —— ⚠️ **它是 `ci.yml` qt-smoke 步驟的顯式清單成員**（`.github/workflows/ci.yml` 的 `Run offscreen synthetic Qt smoke` 步驟內）⚠️ **換檔就必須改 `ci.yml`，而那是本專案反覆踩到的「守護存在但沒有任何 CI job 執行它」的成因。**

⚠️ **重寫範圍（採納 commander 同意的方案）**：

| 部分 | 處置 |
|---|---|
| **五個 helper**：`_has_qt()`／`_gallery()`／`_profile()`／`_desktop()`／`_window()`／`_round_row()` | ⚠️ **全部保留、原地重用** —— 它們是真實 wiring（真 `FakeCapture`、真 `DesktopSession`、真 `_QtResearchWindow`）|
| **七支現有測試** | ⚠️ **全部作廢** —— 它們測的是「兩個每輪下拉可獨立設定」，新 UI 沒有下拉 |
| **新增 `_press_key` 端到端**（= R1）| ⚠️ **必須走真實路徑：按 ✓／✗ → 讀 CSV → 斷言該列的 `probe_kind`／`presenting_identity`／`operator_verdict`** |
| **保留 `test_each_round_can_set_its_own_value` 的時間命題** | ⚠️ 命題仍成立（「每輪可獨立設定」是關於時間的），⚠️ **只是設定方式從下拉改成按鈕** |

⚠️ **⚠️ 新測試必須真的會紅 —— 這是本階段的核心驗收**：

⚠️ **⚠️ 不得只讀斷言文字。** ⚠️ 必須實測：

```bash
# 在 git archive 另開的副本上做突變，不要原地改再 cp 還原
# 突變 1：移除 _press_key 內 append_g3_demo_results_csv 的 operator_verdict 傳參
#   → 新測試必須紅
# 突變 2：把 _press_key 的 writer 整段移除（ast.parse 須仍通過）
#   → 新測試必須紅
# 突變 3：把 operator_verdict 寫成常數而非從按鈕狀態讀
#   → 新測試必須紅（這是 writer_wiring 的 _is_live_read 保護的性質）
```

⚠️ **⚠️ 突變腳本必須加會拋錯的前置斷言**（套用後 `ast.parse` 通過才寫檔），⚠️ **並在套用後先確認它真的套用了**（例如 grep 該 kwarg 命中數從 1 變 0）再跑 pytest。

⚠️ **⚠️ 為什麼這條紀律要寫進計畫**：⚠️ 本專案已發生過四次「機械動作失敗但後續結果看起來正常」⚠️ —— ⚠️ `cp` 回報還原成功但突變殘留（靠「AST 只看到 1 條 writer」的異常回頭才發現）· ⚠️ 突變腳本拋 `StopIteration`（突變根本沒套用）而緊接著的 `3 passed` 來自未突變的乾淨樹 · ⚠️ `gh run list --jq` 缺 `--json` 失敗 → 空 id → grep 得 0 → 回報「守護沒在 CI 跑」（真實值是 2）· ⚠️ `uv sync --extra dev` 不移除既有套件 → 差額計數得假結論。⚠️ **四次裡三次是自己寫的 `assert` 擋下的。**

⚠️ **驗收命令**：

```bash
# 1. qt-smoke job 確實執行它（verify job 只會 skip）
uv run --extra dev --extra research-ui python -m pytest tests/live/test_d7_w0b_probe_kind_input.py -q
# 預期：N passed（無 skipped —— 有 PySide6）

# 2. ⚠️ 對照：verify job 環境下它是 skipped（證明兩個 job 的差額真實存在）
rm -rf /tmp/venv-verify && uv venv /tmp/venv-verify --python 3.14
VIRTUAL_ENV=/tmp/venv-verify uv pip install -e '.[dev]'
/tmp/venv-verify/bin/python -m pytest tests/live/test_d7_w0b_probe_kind_input.py -q
# 預期：N skipped

# 3. 三種突變各自轉紅（見上）
```

⚠️ **review 重點**：⚠️ **新測試是否真的會紅（三種突變實測）** · ⚠️ **五個 helper 是否真的保留（不是重寫成 stub）** · ⚠️ **檔名是否未變**。

---

## 4. 順序強制性與可並行性

| 關係 | 強制？ | 理由 |
|---|---|---|
| P1 → P2 | ✅ 強制 | P2 的 `runbook` 交叉引用指向 spec 的路徑 |
| P2 → P3 | ✅ 強制 | ⚠️ **靜默失敗** —— UI 先做則 `operator_verdict` 不被寫出且看起來正常 |
| P3 → P4 | ✅ 強制 | P3 移除下拉後，現有 5 支測試必然全紅 |
| P2 ∥ P4 | ❌ 可並行 | ⚠️ **但 P4 需要 P3 的新 UI 存在**，所以實際不可並行 |
| P3 ∥ P4 | ❌ 不可並行 | 同上 |

⚠️ **結論：四個階段完全序列，無任何可並行處。** ⚠️ 這是「一個 PR 一個 PR逐筆完成」在技術上的必然 —— ⚠️ **不是流程偏好。**

---

## 5. ⚠️ 階段劃分回報時必須附帶的 CI job 對應表

| 階段 | 新增／修改的測試 | 由哪個 CI job 執行 | 證明方式 |
|---|---|---|---|
| P1 | 無 | — | — |
| P2 | `test_d7_w1_gallery_visibility.py`（改寫）· `test_d7_w0b_writer_wiring.py`（間接）· `test_d7_w0b_probe_kind_input.py`（一行） | **verify**（兩者 PySide6-free）| §1.2 的隔離 venv collect-only |
| P3 | `test_qt_window.py`（既有） | **qt-smoke**（顯式清單）| `--collect-only` ＋ 有 PySide6 的 pass |
| P4 | `test_d7_w0b_probe_kind_input.py`（重寫）| ⚠️ **只有 qt-smoke** ⚠️（verify 只會 skip）| §1.2 的 passed/skipped 差額 |

---

## 6. ✅ commander 已裁決的兩處

### D1 — `writer_wiring` 的 `test_no_writer_passes_a_constant_ground_truth`

⚠️ **該守護比對的是 accessor 的「名字」** —— `_ACCESSORS = frozenset({"_probe_kind_value", "_presenting_identity_value"})`，⚠️ `_is_live_read()` 回傳 `_callee(value) in _ACCESSORS`。

⚠️ **而 spec §4.1 說移除那兩個 accessor 所讀的下拉。** ⚠️ **所以刪下拉 → accessor 必須改寫 → 名字一變這條守護就紅**，⚠️ **而 spec §7.2 第 8 條寫「三條守護全部保留且通過」。**

⚠️ **裁決（commander 2026-10-05）：走 (a) —— 保留 accessor 名字，只改內部實作。**

⚠️ **裁決理由**：⚠️ 那條守護保護的**意圖寫在它自己的註解裡** —— 「『per round』 is enforced only by reading them **at write time**」。⚠️ **「write-time read」在 accessor 內部改讀按鈕狀態之後完全成立**，⚠️ **零測試改動、不犧牲任何保護。**

⚠️ **附帶要求（已寫進 spec §6.2）**：⚠️ **必須在 accessor 的 docstring 寫明「值來自 operator 的按鈕標註，不是 combo 選取」** ⚠️ 否則未來讀者會以為它讀 combo 而誤判。

⚠️ **⚠️ 這條裁決帶出的寫法教訓（已寫進 spec §6.2）**：

> **「守護會不會紅」不能用「斷言大意看起來仍成立」來回答，必須讀比對機制。**

⚠️ **spec §6 初版就是犯這個錯** —— ⚠️ 它寫「守護的 AST 結構斷言仍成立」，⚠️ **而那沒有讀 `_ACCESSORS` 是名字比對這件事**，⚠️ 差點讓整個 plan 建在一個無法達成的驗收條件上（§7.2 第 8 條）。

⚠️ **三條守護的比對對象各不相同** —— ⚠️ **所以只有第三條會因改名而紅**：

| 守護 | 比對對象 | UI 改動後 |
|---|---|---|
| `test_both_demo_row_paths_are_present`（`grep -n 'def test_both_demo_row_paths_are_present'`）| **writer 所在函式名**（`_press_key`／`_record_unlabeled_round`）| ✅ 不受影響 |
| `test_every_writer_forwards_both_ground_truth_columns`（`grep -n 'def test_every_writer_forwards_both'`）| **kwarg 名**（`probe_kind`／`presenting_identity`）| ✅ 不受影響 |
| `test_no_writer_passes_a_constant_ground_truth`（`grep -n 'def test_no_writer_passes_a_constant'`）| ⚠️ **accessor 名**（`_ACCESSORS`）| ⚠️ **改名就紅 —— 這就是 D1 存在的原因** |

### D2 — P3 的 commit 策略

⚠️ **實測過渡狀態**：⚠️ **若先移除下拉、`test_d7_w0b_probe_kind_input.py` 那 5 支會全紅**（它們直接操作 `win.probe_kind_combo`／`win.presenting_identity_combo`）。⚠️ **所以 commander 原本的條件「每個 commit 各自能讓 CI 綠」在移除下拉那個 commit 上不成立。**

⚠️ **裁決：單一 commit ＋ PR 描述分段 ＋ review brief 明寫「請分兩段判斷」。**

⚠️ **「分兩段判斷」的前提**：PR 描述必須**明確切出兩個可獨立判斷的區塊** ⚠️ —— ① 移除舊下拉與改寫 accessor ② 新增按鈕 UI 與 progressive disclosure。⚠️ 否則 reviewer 會當成一個整體改動來看。

⚠️ **additive 骨架方案已被否決** —— ⚠️ 它本身就是 §4.1 說要移除的東西，⚠️ **會產生一個「兩套 UI 並存」的 commit**，⚠️ 那個過渡狀態本身就是一種文件與程式的不一致。

---

## 7. ⚠️ 本計畫**不解決**的事

⚠️ **不要為了讓計畫能排下去而調整 spec 的意圖** —— ⚠️ 以下四項實作時若發現需要解釋，**回報 commander，不自行裁**：
⚠️ ⚠️ **前兩項出自 spec §8.1「本 spec 不解決」**（格子 3 的歸因維度、無效輪漏了誰）。⚠️ **後兩項不是 spec 載明的**，而是本計畫自己的紀律與範圍外待辦：⚠️ `test_capacity.py` 的計時 flaky 不得以 re-run 綠作為通過理由（執行期紀律）、`cli.py` 裡 `grep -n 'ALWAYS empty'` 那行的過期斷言需另開涵蓋 `src/` 的 task（範圍外）。
1. ⚠️ 格子 3 的歸因維度（哪一個 non-target 最容易被誤認）—— 真人測試沒有對照表就拿不回來。
2. ⚠️ 無效輪「漏了誰」—— `invalid_input` + `incorrect` 只有標記，沒有身分。
3. ⚠️ `test_capacity.py` 的計時 flaky —— ⚠️ **本計畫全程不得以 re-run 綠作為通過理由**。
4. ⚠️ `cli.py` 裡 `grep -n 'ALWAYS empty'` 那行的「as the code stands they are ALWAYS empty」在 `#154` 後已過期 —— ⚠️ **不在本 spec 範圍**，需另開一筆涵蓋 `src/` 的 task。