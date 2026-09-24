# G3：本機辨識測試 App

- 日期：2026-09-24
- 狀態：operator 2026-09-24 TUI 直接定義 G3 目標；本文件為 G3 的功能權威。
- 基準：main `42510228b2e09b28a97bfc3370a1ee0be0fc1576`
- 關係：[Phase 2B 研究規格](2026-09-16-phase2b-mac-recognition-research.md) 的研究治理條款（§5 分母細項、§7 觸發表、§8 holdout、執行計畫 §4 manifest 與 §5 R1 預算、參與者／到場／同意帳本）**於 G3 不適用**，保留為日後選用工具，不刪除既有實作。2B 規格附錄 A（方形採集幾何）與 §6 的 truth 隔離在 G3 仍適用。

## 1. 目標

operator 在 Finder **點兩下**開啟本機 Qt App → 相機持續開著 → 有人走到鏡頭前就自動開始辨識 → 5 秒內認出就顯示是哪一張註冊照，認不出就顯示「找不到此註冊人員」→ operator 按「正確／錯誤」→ 自動進入下一輪。

每輪結果存在本機。operator 自行離線測試，**不需要 agent 在場**；測完再請 AI 一次讀資料分析。

G3 回答的問題只有一個：**這支 App 對著真人，能不能認出正確的註冊照。** 不宣稱準確率、不宣稱可部署、`matched` 不等於 `authenticated`。

## 2. 使用流程（行為契約）

1. **開啟**：雙擊 launcher 即開啟視窗，不需要輸入任何終端指令。macOS 第一次的相機權限彈窗由 operator 同意。
2. **啟動檢查**（任一失敗：視窗以中文顯示原因並停在該畫面，不 crash、不靜默繼續）：
   - 模型檔存在且 SHA 相符。
   - gallery 由 `註冊組` 資料夾建成，**每張註冊照一個身份**，身份名稱＝檔名去副檔名（例如 `enroll-23`）。任一張不是恰好一張臉 → 顯示是哪一張。
   - **相機由 operator 從下拉選單選擇**：不論偵測到 1 台或多台，都列出讓 operator 選。能取得名稱就顯示名稱（例如 macOS `system_profiler` 列出的相機名稱），取不到就只顯示編號。**有相機就不得拒絕啟動**；一台都沒有才顯示「找不到相機」。選到非內建鏡頭的後果不處理（最壞是 App 關閉），由 operator 自行選對。不做自動選相機，也不預設選任何一台。G3 App 不使用 issue #89 的 uid／shape 斷言（既有程式保留，不刪除）。
3. **待機**：顯示預覽與方形框，狀態文字「請站到鏡頭前」。
4. **開始一輪**：偵測到人臉即開始。沿既有有界 session：5 秒截止、相鄰取樣至少 200ms、至多 26 幀、時間一致性支持 3 次。
5. **結果**：
   - 認出 → 立即顯示註冊照名稱與分數，不必等滿 5 秒。
   - 5 秒內未認出（timeout／unknown）→ 顯示「找不到此註冊人員」。
   - 其他（多臉、錯誤）→ 顯示簡短原因。
6. **標註**：顯示「正確」「錯誤」兩鍵，由 operator 按下。**標註只來自按鍵，不得取系統預測當答案**；標註不回流辨識。按下後回到待機，進入下一輪。
7. **結束**：關閉視窗即結束並釋放相機。重開後先前資料仍在。

## 3. 資料

- **位置**：store 與 key 分開存放於 `~/Downloads/face_sample/_facecore/` 下的兩個子資料夾。實際路徑寫在該資料夾內的本機設定檔，不進 Git。不得放在 Desktop／Documents（本機已開 iCloud 桌面與文件同步）。
- **每輪一筆紀錄**：輪次 ID、開始時間、結果類別、顯示的身份、top1／top2 身份與分數、margin、耗時、取樣幀數、operator 標註、profile 版本、模型與 gallery digest。
- **影像**：沿用既有加密存幀（每輪至多 26 幀），供事後分析失敗原因。
- **診斷**：每幀的真實偵測與品質值（信心、臉框、landmark、清晰度、亮度、角度、拒絕原因）必須寫入 trace，不得是佔位值。
- **彙整檔**：另輸出一份不含影像、不含 embedding 的 `results.csv`（每輪一行），operator 可直接打開看。
- **保存期限**：G3 的影像與紀錄保存 30 天，避免 operator 還沒找 AI 分析，資料就先被自動清掉。

## 4. 起始 profile `g3-v1`（凍結，放 repo）

| 欄位 | 值 | 來源 |
|---|---|---|
| match_threshold | 0.363 | OpenCV Zoo SFace 上游預設 cosine 門檻（`models/face_recognition_sface/sface.py` 的 `_threshold_cosine = 0.363`） |
| margin_threshold | 0.10 | 專案既有設計錨點（`src/facecore/cli.py:285`），不是從任何 sweep 結果挑選的 |
| review_threshold | 0.30 | 只影響顯示分帶，須 ≤ match |
| timeout_ms／sample_interval_ms／max_frames／queue_limit | 5000／200／26／1 | 既有研究預算（`d-20260920132145277296-1`） |
| required_support／min_support_interval_ms | 3／200 | 既有候選臂起始規則 |
| continuity_max_center_delta_ratio | 0.50 | 既有初值 |

G3 期間不因為看了結果就調整門檻。要改就發新版本，並經 operator 同意（屬 G4）。

## 5. G3 不做

多人入鏡停止（測試環境由 operator 確保只有一人）、參與者／到場／情境 ID、同意帳本、實驗 manifest 八區塊、holdout、T04–T12 觸發器、雲端同步檢查、撤回 CLI、headless 方形裁切、自動選相機、簽章 `.app`／pyinstaller 封裝。

## 6. 仍適用的底線

- 照片、embedding、資料庫、權重、逐輪結果都不進 Git，也不上傳雲端。
- 辨識不寫入 gallery、不學習。
- 標註不回流 scorer。
- 對外文字不貼完整相機 uid。

## 7. 驗收

1. Finder 雙擊 launcher 即可開啟（首次權限彈窗除外）。
2. 模型、gallery 兩類啟動失敗與「找不到相機」都顯示中文原因；有 2 台以上相機時出現下拉選單，且不會因為多台相機而拒絕啟動。
3. 能連續多輪：成功或失敗、按鍵之後自動進入下一輪，相機不需重開。
4. 每輪有一筆紀錄加標註；重開 App 後資料仍在；`results.csv` 與加密 store 的輪次一致。
5. 標註只來自按鍵；有測試證明誤認時可以標「錯誤」，不會自動記成正確。
6. trace 的診斷欄位是真值，有測試證明不是 None 佔位。
7. 自動測試只用 synthetic／test double。真機驗收由 operator 依 SOP 自行執行，agent 不開相機。
8. 附一頁中文 SOP：怎麼開啟、畫面各狀態的意思、怎麼測、資料在哪裡、測完要跟 AI 說什麼。
