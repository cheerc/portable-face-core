# G3：本機辨識測試 App

- 日期：2026-09-24
- 狀態：operator 於 2026-09-24 回覆「ok」接受設計，並指示直接交 lead；**尚未落 repo，不表示程式已修好**。目標：覆寫 repo 的 `docs/specs/2026-09-24-g3-local-test-app.md`，不另立第二份功能權威。
- 修訂基準：main `6b058fd8335581bba644850f91b93a381482390c`；方向裁決 `d-20260924080117997753-5`；真機 RCA `t-20260924074541870720-80684-50`。
- 與 [Phase 2B 研究規格](2026-09-16-phase2b-mac-recognition-research.md) 的關係不變：§5 分母細項、§7 觸發表、§8 holdout、執行計畫 §4 manifest／§5 R1 預算、參與者與同意帳本於 G3 不適用；附錄 A 方形採集幾何及 §6 truth 隔離仍適用。

## 1. 目標

operator 在 Finder 雙擊啟動本機 Qt App → 從下拉選單選相機（**鏡頭仍關閉**）→ 按 **Start** 才開鏡頭並開始一輪辨識 → 辨識結束即關鏡頭、清空照片預覽、顯示文字結果 → operator 按「正確／錯誤」存檔 → 清空上一輪結果，保留已選相機，待下一次 Start。**不會因選相機或按標註而自動開鏡頭。**

operator 可自行離線測試，過程不需 agent；結果留本機，測完後才請 AI 分析。這只回答「本人站到相機前時，本機原型能否認出正確註冊照」，不宣稱部署準確率或身份認證；`matched` 不等於 `authenticated`。

## 2. 操作與相機生命週期

1. **啟動／準備**：雙擊 launcher 即可開 Qt 視窗。先檢查模型雜湊與 `註冊組` gallery；每張註冊照一身份，身份名稱為檔名去副檔名。列出可選相機；沒有相機時顯示中文原因。相機名稱取不到時顯示編號。有相機就不因多台而拒絕。**列舉和選擇不得開啟視訊串流、取得影格或建立待機預覽**；沒有選相機前 Start 停用。不自動選相機，也不使用 issue #89 的 uid／shape 斷言。
2. **待開始（Ready）**：保留選定相機；相機未開，預覽區為空白／「相機未啟動」佔位文字，沒有先前人臉畫面。Start 可按，「正確／錯誤」停用。不在此狀態建立影像 bundle 或開始 5 秒倒數。
3. **Start／辨識中（Running）**：operator 按 Start 後才對所選相機 `open`；成功開啟後立刻顯示持續更新的視訊預覽與方形引導框，開始一輪 5 秒上限辨識。5 秒由**本輪成功開鏡頭後**的單調時鐘起算，不從選相機、啟動 App 或前一輪開始；macOS 權限等待與模型預備時間不得消耗辨識窗口。相鄰取樣至少 200 ms，至多 26 幀；同身份支持規則沿用凍結 profile。正常 matched 可在窗口到期前結束，不能等滿 5 秒才顯示。UI 倒數跟本輪真實時間前進；一次 Qt timer tick 的工作步數耗盡**不是** 5 秒截止或失敗判據。
4. **終局／結果（Result）**：matched、到時 unknown／timeout、invalid_input、讀相機失敗、Cancel／視窗關閉都停止本輪並釋放相機與讀取 worker。對會顯示結果的終局，**先確認鏡頭已關閉，再清空 QPixmap／預覽快取，只顯示文字結果及可用按鍵**；結果畫面不保留上一幀人臉。matched 顯示註冊照名稱及分數；有合格影格但 5 秒內未認出才顯示「找不到此註冊人員」。若完全無影格、無可用臉或全部品質拒絕，顯示對應的「未取得可辨識影格／相機讀取異常」及已知原因，不要冒稱已驗證不在 gallery，也不要把 `zero_usable_frames_collected` 當成認錯人。
5. **標註與下一輪**：結果顯示後由 operator 按「正確」或「錯誤」；按鍵標註與加密紀錄、`results.csv` 的同一輪資料必須成功寫入，失敗則留在結果畫面顯示錯誤、不開下一輪。成功後清除身份結果、預覽與暫存影格，回到 Ready；**原相機選擇保留但鏡頭關閉**，必須再按 Start 才有下一輪。按鍵不能自動把模型預測當真值，也不能回流 scorer。
6. **Cancel／關窗／錯誤**：Running 按 Cancel 應停止並關鏡頭、清空畫面，回 Ready；不偽造成功的辨識結果。任一相機 `open`／`read`、model／gallery 或 store/key 錯誤都以中文指出所知原因、釋放資源，不無限重試或假裝 unknown。任何時候關視窗皆釋放鏡頭；已標註完成的輪次保留，未標註輪次不作已完成測試。錯選相機最多使這一輪失敗，不得靜默切換其他相機。

## 3. 每輪資料與訊息真實性

- store 與 key 分開存於 `~/Downloads/face_sample/_facecore/` 下的本機設定位置；照片、embeddings、DB、權重、逐輪結果均不進 Git、不上傳雲端。影像與紀錄的 G3 保存期限為 30 天。
- 每輪紀錄含輪次 ID、時間、結果類別、顯示身份、top1／top2 身份與分數、margin、耗時、實際取樣／合格／拒絕幀數、operator 標註、profile、模型與 gallery digest。`results.csv` 每個已標註輪次一列；無論結果是 matched 或失敗，都不丟掉已發生的錯誤資訊。
- 每一**實際辨識輪次**的影像 staging、crop mapping、真實診斷與 trace 都綁到該輪的 attempt/session；預覽必須與該輪推論使用**同一組方形幾何映射**。開始前的設定／前一輪資料不得被當成當輪資料；一輪失敗不得覆寫另一輪。只在完整 commit 成功後才顯示「已保存」，crop mapping 單獨持久化不算結果已存檔。
- 多臉特殊演算法、校準門檻、改註冊照、自動相機選擇、learning／promotion、holdout 與正式帳號畫面均不在此修訂範圍。仍由 operator 在單人背景測試。gallery 固定 one-shot，使用既有 `g3-v1` profile；本輪不根據第一次真機失敗調參。

## 4. 原因與修復範圍

首次 operator 真機測試在相機選定後出現靜止預覽、倒數維持 `5000 ms`，結果 `invalid_input：zero_usable_frames_collected`。原碼盤點證成三個必修結構缺口：初始 `DesktopSession` 有預覽／裁切 callbacks，連續 round 新建 session 卻漏接；Qt 倒數 closure 綁初始 session；非固定窗口在單個 tick 用完步數就提早 `finish`。此因果只說明**可重現的程式路徑**，不聲稱已從真人影像確認無合格幀的唯一原因。

修復應延用一條辨識 pipeline 與現有 Qt／session／recorder，處理**每輪**的相機所有權、preview、crop、staging、trace、計時及 close-out。初始 session 的 callback 寫死其 attempt id，recorder 原有 append-frame 只支援單一 active staging；不能直接把舊 callback 物件搬到新輪次就宣稱完成。不中斷既有 headless／checkpoint 研究路徑。

## 5. 驗收（synthetic 先行，真機最後）

1. 從真正的 `live --ui qt` 啟動路徑（硬體只在最底層用 FakeCapture 隔離），證明視窗出現及選相機後 **camera open/read 次數皆為零**，沒有真人影格或預覽；Start 前可更換相機。按 Start 後只開所選那台一次，預覽在 running 的連續不同影格上更新、倒數從本輪開始前進。不得只對單獨的 Qt helper 測試。
2. 同一條整合路徑驗 matched 提前結束與 5 秒未匹配兩種結果；到時須由時鐘決定，50-step/單個 UI tick 用盡不能提早終結。全無影格、沒有臉、全部品質拒絕三類可診斷區分；正常短暫讀取空缺不能冒充未註冊者。
3. 每種終局（matched、timeout／unknown、invalid_input、相機讀取失敗、Cancel、關窗）皆驗證鏡頭及 reader 釋放，**結果顯示時 preview pixmap 已清空**；標註後 Ready 保留相機選項、鏡頭仍關、上一輪身份與照片均清空。下一輪必須重新 Start 才有 open/read。
4. 一次完整雙輪：同一選相機，Start→結果→標註→Ready→Start→結果→標註；每輪 crop mapping／加密影像／trace／label／CSV 指向**自己的** attempt/session，且成功標註後立即可讀、重新開 App 仍存活。任何一輪寫入失敗不得顯示「已保存」、不得自動開始下一輪；不遺漏或重複計數。
5. 測試不讀真人 corpus 或開真相機，先用可觀察的 RED 證據讓修前版本在上述整合情境失敗，再修到 GREEN；operator 最後自行依更新 SOP 真機驗收。SOP 必須反映「選相機不開、Start 才開、結果關、標註後照片清空、下輪再按 Start」。
