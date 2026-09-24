# G3 本機辨識測試 App：執行計畫

拋棄式執行計畫：做完即不維護；功能權威是 [G3 spec](../specs/2026-09-24-g3-local-test-app.md)，兩者衝突時以 spec 為準。基準 main `42510228b2e09b28a97bfc3370a1ee0be0fc1576`。

## 原則

G3 只要一件事：operator 點兩下開 App，對著鏡頭，看它認不認得出來。凡是不影響這件事的工作都不做。

## 工作包（由 lead 決定如何切 PR）

| # | 內容 | 修的是哪個已知缺口 |
|---|---|---|
| W1 | `g3-v1` profile 檔進 repo（spec §4） | 無可追溯 profile |
| W2 | Qt 連續模式：待機 → 偵測到臉即開始一輪 → 顯示結果 → 正確／錯誤 → 回到待機；相機全程不重開；認出即顯示，不等滿 5 秒 | 現行 Qt 一次只跑一輪，Start 在視窗顯示前就被呼叫（`research/cli.py:752` vs `:779`） |
| W3 | 標註改為 operator 按鍵「正確／錯誤」，逐輪落盤 | enrolled 標註取系統預測（`live/qt_window.py:425-427, 460-466`） |
| W4 | 每輪紀錄＋`results.csv`；G3 保存期 30 天 | 同意／TTL 寫死（`research/cli.py:426-435`） |
| W5 | 真實 `FrameDiagnostics` 寫入 trace | 診斷收集後被丟棄（`research/cli.py:585-592`；`research/diagnostics.py:70-115`） |
| W6 | gallery 直接由 `註冊組` 資料夾建成，身份＝檔名；本機設定檔提供資料夾、模型、store／key 路徑；相機改為啟動時由 operator 從下拉選單選擇（不用 uid／shape pin） | 現行需手寫 manifest，且路徑檢查 `repo_root=models`（`research/cli.py:297`）；多相機時 uid 斷言會拒絕啟動 |
| W7 | 雙擊 launcher（例如 `.command`），讀本機設定檔；啟動檢查失敗時顯示中文原因 | 每次都要 agent 手動組指令 |

**不做**：見 spec §5。

## 未解前提（lead 開工前請先確認）

1. canonical checkout 在 merge 後怎麼更新到 main、`uv` 環境怎麼備妥，launcher 才能在 operator 的機器上直接執行。
2. 用 `.command` 從 Finder 啟動時，macOS 相機權限會記在 Terminal 名下：請確認第一次彈窗與後續行為，並寫進 SOP。
3. 設定檔放 `~/Downloads/face_sample/_facecore/`，repo 內只放範本。
4. 下拉選單的相機名稱要怎麼對到 OpenCV 的 index，請先確認現有列舉程式（`src/facecore/live/camera_identity.py`、`scripts/verify_camera_identity.py`）可以重用到什麼程度。對不準時退回只顯示編號，不要為此另外開發。

## 順序

W1–W7 merge → commander 寫 SOP（由 lead 落 repo）→ operator 依 SOP 自行測試（agent 不參與）→ operator 通知 → AI 讀資料分析。

真機的第一次開啟就是 operator 的 smoke。若失敗，operator 回報畫面上的訊息即可。
