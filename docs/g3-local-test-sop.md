# G3 本機辨識測試：操作 SOP

**狀態：第一次真機測試待 operator 自行執行。** 本文件依 `cheerc/portable-face-core` main `46cd44b` 撰寫。48 個 G3 自動測試全過；以合成影格在背景模式走完「開視窗→選相機→兩輪辨識→按正確／錯誤→即時存檔」；本機真模型已成功由「註冊組」建出 gallery。**尚未由 operator 在真相機上驗過。** 不要把此文件當成真人辨識已成功的證據。

## 首次使用前（只做一次）

- 程式要先更新到含 G3 App 的版本。若 `~/portable-face-core/scripts/g3-local-test-app.command` 尚不存在，請先在 Terminal 執行：
  ```sh
  git -C ~/portable-face-core pull --ff-only origin main
  ```
  如果更新失敗，停下並把 Terminal 訊息交給團隊處理；**不要刪除資料夾或用 `git reset --hard`**。之後每次啟動，launcher 會自行嘗試快轉更新；離線或更新失敗時會用現有 checkout 繼續。
- 本機設定檔已備妥：`~/Downloads/face_sample/_facecore/g3-local.json`，指向 `註冊組` 的 23 張照片及 `~/facecore-models` 的模型。**不要把設定檔、照片或結果提交到 Git。**
- 第一次啟動會執行 `uv sync --extra research-ui` 備妥環境，可能需要網路及一些時間。之後仍會檢查環境；這是目前 `.command` 的行為，尚未保證完全斷網也能啟動。
- 找乾淨、只有一人入鏡的測試背景；如果請別人測試，由你在 App 外取得對方同意。目前不處理排隊時的多臉畫面。

## 怎麼開與怎麼測

1. 在 Finder 按 **⌘⇧G**，貼上 `~/portable-face-core/scripts`，找到並**雙擊** `g3-local-test-app.command`。Terminal 會開啟，等它準備好後會出現 Qt 視窗。第一次若 macOS 詢問相機權限，請依你的意願允許 **Terminal** 使用相機；若未見視窗，先讀 Terminal 裡的訊息。
2. 視窗中的相機下拉選單起始是「請選擇相機」。**由你選內建相機**；若只能顯示編號，可逐一確認。**選相機時鏡頭不會開、沒有預覽**，這是正常的。程式不會自動選，也不會因有 iPhone 相機就拒絕啟動。若選錯而失敗，關閉視窗再開即可。
3. 選好相機後按 **Start** 才開鏡頭並開始一輪：畫面出現即時預覽與方形框，倒數從本輪開始走。認出時顯示例如 `enroll-23 0.62`（可提前結束，不必等滿 5 秒）；5 秒內未認出顯示「找不到此註冊人員」；若完全沒影格、沒人臉或品質全被拒絕，會顯示對應的中文原因（不是「找不到」）。分數是相似度，不是正確率。畫面上的身份是註冊照片檔名（去掉副檔名），**不是你的帳號**。
4. 辨識結束**先關鏡頭**：結果只留文字，照片已清空。請按一次：
   - **正確**：顯示的是實際對應的註冊照；或測試者原本未註冊而顯示「找不到」。
   - **錯誤**：你有註冊卻顯示別人／找不到，或未註冊者卻被認成任何一張註冊照。
   按完即把這輪寫入本機（右下「已保存」只在完整寫入後才亮；若顯示「紀錄寫入失敗」就停下回報，不要繼續）。接著回到待開始：**相機選項保留、鏡頭仍關、上一輪文字與照片都清空**，下一輪要再按一次 Start。若你只有自己測，`enroll-23` 才是正確的身份。
5. 測完用**視窗的關閉鈕**結束（任何時候關窗都會關鏡頭）。沒有按「正確／錯誤」的最後一輪不算已完成測試；已按過鍵的輪次已即時寫入，即使後來 App 意外結束也不應丟失。

## 目前是哪一種模式（D3 起）

雙擊 `g3-local-test-app.command` 開啟的 App 是 **demo 模式**（`--mode demo`）：**不保存任何影像、embedding 或逐幀診斷**，按下「正確／錯誤」時只把這一輪寫成**明文**一列。這是為了讓一般辨識不必留下生物特徵資料。

| | demo 模式（雙擊 App，預設） | record 模式（研究執行器） |
| --- | --- | --- |
| 怎麼開 | 雙擊 `g3-local-test-app.command` | Terminal 執行下方命令 |
| 影像／embedding | **不寫任何** | 加密寫入本機 |
| 逐輪結果檔 | `store/demo-results.csv`（明文） | `store/results.csv`（明文，固定 17 欄） |
| attempt ledger | 不寫 | 寫入 |
| 兩個同意勾選 | **不需要**（帶了會直接報錯，不會靜默忽略） | 必須兩者都帶 |

研究錄製（record 模式）要用的命令：

```sh
uv run --extra research-ui python -m facecore.research.cli live \
  --profile profiles/g3-v1.json \
  --store "$HOME/Downloads/face_sample/_facecore/store" \
  --key-dir "$HOME/Downloads/face_sample/_facecore/research_keys" \
  --device 0 --session "g3-$(date +%Y%m%d-%H%M%S)" \
  --mode record --record-consent --image-consent \
  --ui qt --continuous --config "$HOME/Downloads/face_sample/_facecore/g3-local.json"
```

`--record-consent` 與 `--image-consent` 在 record 模式缺一即拒絕執行。

## 資料在哪裡、怎麼交給 AI

- `~/Downloads/face_sample/_facecore/store/demo-results.csv`：**demo 模式（雙擊 App）**的逐輪摘要，可用試算表打開。欄位含模式、App 版本、profile 版本、gallery digest、失败原因代碼、顯示身份、top1／top2、分數、差距、有效幀／所需幀、標註與時間。**這是明文，且不含任何影像或 embedding**。
- `~/Downloads/face_sample/_facecore/store/results.csv`：**record 模式**的逐輪摘要（固定 17 欄）。demo 模式**不會**寫這個檔，既有內容與修改時間都不變。
- `~/Downloads/face_sample/_facecore/store/`：record 模式下的加密影像與逐幀診斷。解密鍵在同層的 `research_keys/`，**不要隨意複製、上傳或分享這兩個資料夾**。demo 模式不會在這裡留下加密資料。
- CSV 裡「錯誤」的 `label_kind` 目前記作 `uncertain`：這代表你按了「錯誤」，**不是系統已知道正確身份**。事後請 AI 依你實際測試的人和畫面結果解讀，不要直接把 `uncertain` 當作「未註冊」。
- 影像與紀錄設定的保存期限為 **30 天**；不要等到期限到了才請 AI 分析。可以在本機對 AI 說：「我完成 G3 測試。請先唯讀分析 `~/Downloads/face_sample/_facecore/store/demo-results.csv`（若我用的是研究錄製則是 `results.csv`），告訴我每輪是否正確、錯誤型態與下一步；**不要開相機、修改資料、把照片上傳或拿測試結果調門檻**。」

## 卡住時

| 情況 | 先做什麼 |
|---|---|
| 雙擊後沒開視窗 | 查看 Terminal；首次更新、`uv` 環境、設定檔或模型／註冊組錯誤可能在**Terminal** 顯示，而非 Qt 視窗。把**去除個人路徑與完整相機識別碼**後的錯誤訊息交團隊。 |
| 「找不到相機」、下拉選單空白 | 確認 macOS「系統設定 → 隱私權與安全性 → 相機」已允許 Terminal。關閉其他佔用相機的 App 後重開。 |
| 按 Start 沒反應 | 先確認已選相機（「請選擇相機」會擋下 Start）與兩個同意勾選；相機被別的 App 佔用時會顯示中文原因，關掉佔用的 App 再按一次。 |
| 註冊組建構失敗或模型載入失敗 | 不必自己更換照片或模型；保留 Terminal 錯誤供團隊判斷。不要把照片傳進對話。 |
| 找不到結果 | 先確認這輪按了「正確／錯誤」，再看 `store/demo-results.csv`（雙擊 App）或 `store/results.csv`（研究錄製）。標註按鍵後若畫面顯示「紀錄寫入失敗」，不要繼續下一輪，回報錯誤。 |

**界線：** 這是本機原型的單人測試，不是認證、考勤、部署或多人排隊驗收；分數與少量試驗不能證明真實誤認率。首次真機操作的結果（包含出現的權限彈窗與相機名稱）仍需你實測回報。
