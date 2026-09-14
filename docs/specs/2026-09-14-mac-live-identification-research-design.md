# Phase 2A：Mac 動態辨識研究原型設計

- 日期：2026-09-14
- 狀態：operator 已同意方向並授權整理正式設計／更新文件；本文件待正常 review／merge。尚未授權產品實作或開始蒐集影像。
- 上位規格：[Portable Face Core Design](2026-09-09-portable-face-core-design.md)；階段決策：[ADR 0008](../decisions/0008-mac-live-identification-research.md)。
- Phase 2A 是新增的研究子階段，不是 Android／iOS 開發，也不是模型選型或 Phase-1B 全面驗收通過的宣告。

## 1. 目的與成功定義

讓操作者在 Mac 開啟相機，沿用每人一張註冊照，邀請本人做動態掃臉；同一工具完成拍攝引導、有限時間內的多幀辨識、結果標記與經同意的回放資料保存。資料隨實驗累積，不要求先手工建立大型圖庫才能開始。

第一個可交付成果：一次掃臉有明確開始／結束、品質提示、唯一終局結果與可追溯版本；固定 gallery 下可重播相同 session，比較品質最佳單幀與多幀策略。沒有認出時安全結束，不強行猜名字。

原型成功是研究流程可用且結果可重現，不是先承諾某個準確率。模型、前處理、取樣、引導、多幀策略各自可替換，使用同一批封存 session 比較。單張靜態失敗不直接推論動態失敗；動態改善亦須由 session 級結果證明。

## 2. 範圍與非目標

第一版納入：Mac 內建或 USB 相機、本機視窗、固定 one-shot gallery、單人引導、多幀 session、離線回放、評估標記、可選的受控抽樣影像保存。相機與 UI 不進核心辨識 policy。

排除：Android/iOS、遠端相機、HTTP/REST 或服務部署、身份同步、考勤／門禁、多人身份追蹤、無人值守運行、foundation-model 訓練、自動更新正式模板庫、活體與防翻拍承諾、`authenticated`。全身框僅為構圖提示，實際品質依臉部框與 landmark 判定。

推薦薄本機桌面視窗而非瀏覽器＋服務，避免為研究引入網路傳輸／server 生命週期。Python 核心與現有 ONNX 工具沿用；UI／capture library 必須在 implementation plan 的有界 spike 中確認 macOS 權限、打包、license、主執行緒限制與關閉行為，未實測前不指定已可用套件。

## 3. 使用流程與 session 邊界

1. 載入既有單張註冊照所建立的研究 gallery；記錄 gallery revision/digest、模型與前處理版本。開 session 前凍結 gallery，整次 session 不允許背景換模型或模板。
2. 操作者明確開始本次嘗試；顯示相機權限、錄製開關與研究模式。實際身份不送辨識器；評估標籤在結果鎖定後由操作者填入或於隔離的 evaluator 讀取。
3. 預覽引導置中、距離、光線、清晰度與姿態。預覽可鏡像，但推論影像 orientation/color contract 不可暗中改變。
4. 單臉且品質通過才累積證據；零臉、多臉、相機斷線、時間倒退、明顯位置跳躍／人物切換均清空窗口。第一版不跨遮擋追蹤人物；無法確認連續性則要求重新開始。
5. 成功、逾時、取消、輸入錯誤或硬錯誤皆產生一次終局，之後不再計數；新嘗試須再次開始，禁止長時間無上限試到偶然通過。
6. 操作者標記真實身份／未註冊者／不確定，以及可選情境。標記錯誤可更正但留下版本；未標記 session 不進有監督準確率分母。

擬定研究 profile：每 session 最長 5 秒、推論取樣上限每秒 5 幀、最新幀優先、佇列上限 1；此為待 spike 驗證的工程預算，不是已選的辨識水位。profile 必須存入每次結果，修改後產生新版本；禁止依單次結果自動延長 session。

## 4. 元件、資料流與可重用邊界

- **Capture adapter**：輸出 timestamp、frame sequence、尺寸、orientation、RGB 像素；preview 與 inference 分離。drop 舊幀而非排隊處理過時畫面，Stop/視窗關閉必須釋放裝置和 worker。
- **Frame pipeline adapter**：重用 `pipeline/yunet.py`、`align.py`、`measure.py`、`quality.py`、`embed.py`。現有 `decode_image` 是 encoded-bytes 入口；已解碼相機幀的入口需新增並以既有前處理 conformance 驗一致，不能默認 ndarray 已符合色序／鏡像／旋轉規則。
- **Single-frame scorer**：重用 `policy/identify.py` 與 repository contract，產生每身份分數、品質、top candidates、margin、版本。除錯排名與使用者終局分離，uncertain 不顯示猜測名字。
- **Session decision engine（新增）**：純時間有序 observation 輸入；不讀相機、不持有檔案路徑、不接受 ground truth。輸出 session 狀態、支持幀摘要與終局結果。
- **Research recorder/evaluator（新增、獨立）**：記錄結果與標籤、執行同 session 回放與策略比較；不直接寫正式模板庫。
- **UI adapter（新增）**：預覽、引導、Start/Cancel/標記/保存/刪除；顯示相似度非身份認證。

現有 `eval/real_replay.py` 是評估 harness，不當作線上追蹤或 session engine。既有 `IdentificationResult` 不直接塞入不相容欄位：session envelope 另版本化，包含 session_id、終局原因、耗時、有效／拒絕／丟棄幀數、profile、模型與 gallery 版本；精確 Python signatures 由後續 plan 定義。

## 5. 多幀策略與對照實驗

第一版必須有兩條可回放對照：

- **品質最佳單幀基線**：在相同有界 session 中依品質而非身份分數選幀；不能挑「最像某人的一幀」美化基線。終局只在 session 結束產生。
- **時間一致性候選策略**：相同身份在有間隔的合格幀中獲得支持，同時通過 score 與 margin；拒絕 `any-frame-wins` 和只看最大分數。研究起始 profile 可測 3 個至少相隔 200ms 的支持幀；新合格幀支持不同身份時清空支持窗口，None margin 或單身份無 runner-up 不構成成功證據。

穩健分數融合（median/trimmed mean）列為後續候選，不是第一版必要元件。多幀 score/margin 門檻採 versioned 實驗 profile，延用單幀門檻作對照不等於完成 session 校準；不得把靜態 M5 的 0.15 直接寫成新的產品預設。

相鄰幀高度相關；幀數不是獨立樣本數。成功判定必須同時受 session 時限與重試規則約束。每種策略都在同一封存輸入序列上重跑；同時報告固定窗口結果與提前成功耗時，不能用不同長度資料假裝公平比較。

## 6. 記錄、同意、保存與刪除

預設 camera frames/crops/embeddings 只在有界記憶體窗口，Stop、取消、逾時、錯誤後清除參照；不宣稱 Python/OS 可保證物理記憶體清零。關閉影像保存仍可測辨識，但不能宣稱可換 detector/embedder 重播。

兩類不同的資料與權限：

1. **評估紀錄**：opaque participant/session IDs、ground truth（含 unknown/unlabeled）、終局、耗時、品質／分數摘要、配置及版本。它們仍可能是敏感個資，不稱為匿名；加密保存，原型預設 TTL 30 天，啟動及寫入前清理到期資料。姓名對照分開管理；無 raw embeddings、照片、絕對路徑進一般 log。
2. **研究抽樣影像**：獨立明確同意後才啟用。每次 session UI 顯示保存狀態；僅保存上述取樣幀（上限 25 張、5 秒），不錄音、不連續背景錄影。預設 TTL 7 天。為重跑 detector，需要有限全幀；這是研究 recorder 的明列例外，不是 Face Core exemplar 政策變更。使用單人受控背景；觀察到多人即停止保存並丟棄本次尚未提交的影像 bundle。不得宣稱偵測器能發現所有背景旁人，仍須實驗場地控制。

上述 TTL 是正式設計的研究預設、非現有實作事實；資料收集開始前須在 consent UI 顯示並確認目的、資料種類、期限、刪除方式，涉及未成年人須先完成適用監護授權。文件核准不是個別參與者同意。

研究 bundle 在 repo 外透過 AEAD 加密直接寫入；key 與資料分離，不使用 plaintext temp、預覽截圖、crash dump 或雲端同步作旁路。缺 key／空間不足／加密或完整性失敗：停止錄製，明示 session error，不退回明文。相機預覽與檔案寫入均不得在同意前啟動研究保存。

使用 per-session key 或可證明等效的刪除機制；關聯 labels、索引、衍生 crops/embeddings、已知本地匯出一併刪除。原型不自動備份或外傳；session 刪除須可跨重啟驗證，未知／失去管理的副本不能宣稱已刪。到期後 application 關閉期間不執行清理，但重啟先清理再開放讀取。整批磁碟加密不是 app-level AEAD 的替代品。

研究同意不是 learning confirmation：即使保存 negative/unknown 回放樣本，也不得生成 candidate。`not_me` 在既有 Face Core 學習路徑仍不保留觀察；只有這個獨立、已同意、有限研究 recorder 可以保存評估素材。

## 7. 受控學習的分階段安排

第一版固定 gallery、只讀正式庫或從單照建立隔離研究 gallery；不呼叫線上 promotion。記錄每輪 gallery digest，證明測試期間未更新。

後續離線學習研究須另定 task：baseline 與 adaptive 各有隔離的研究 repository，僅以先前 session 的明確確認生成候選；同次視訊最多是一個獨立事件，保留 1B burst suppression、跨身份 exclusivity、generation、rollback 與 no-future-leakage。未來 session 評估學習效果，禁止同段影像一邊學、一邊當測試。不得為得到 creation 任意下調 0.88，也不以模型自己的成功取代確認。

## 8. 評估單位與報告

以 session 為主要分母；分別報告本人正確認出、本人認錯其他已註冊身份、未註冊者誤接受、review/unknown/timeout/invalid/error/cancel、time-to-decision（含失敗／逾時，成功樣本另列）、品質拒絕、推論延遲與丟幀。必須同時列 attempted、labeled、eligible、successful 分母，不能隱去失敗 session。

先用現有註冊人員做少量 normal/光線/姿態實驗及同意參與的未註冊者。按 session／到訪切分 development 與 sealed holdout，不按相鄰幀隨機切分。任何 holdout 一經用來調整 profile 即不再是 holdout；增加新的後續 session 驗證。要作跨身份泛化主張，還須身份層級隔離；首輪只報現有 gallery。

新增模型須通過原有 license/provenance/integrity gate，再用同張初始註冊照、同 session、同標籤與相同 profile 比較；新模型重算 gallery generation，不混向量。圖庫逐步累積，不先設大型人數要求作研究啟動門；正式多身份部署／500 人宣稱仍須代表性圖庫重校準。

## 9. 驗收矩陣與失敗處理

- fake camera + synthetic frames 驗 Start/Stop/Cancel/timeout、零臉／多臉／切換清空窗口、無限取樣不得延長期限；同 session 最多一個終局。
- 同幀不同鏡像／orientation/color 表示經 adapter 後要符合既有 conformance；model mismatch 拒絕處理。
- 分數邊界、None margin、身份交替、一個高分尖峰、重複 timestamp／sequence、stale frames 都不能意外成功；deterministic replay 重複輸出一致。
- labels 缺席或改寫不影響 scorer/session 結果；僅改評估分類；任何未註冊 false accept 都進報表。
- 無 consent 不產生影像檔；中途撤回清除未提交 bundle；有 consent 的 negative 可保存但不學習。驗 TTL、key failure、disk failure、刪除後重啟、無 plaintext temp、無網路送出。
- 相機 permission denied/disconnect、worker failure、UI close 皆完成有限清理，不留下相機佔用或背景 process；camera error 不顯示 matched。
- 真機人工 smoke：一位已註冊與一位已同意未註冊參與者；測試程式輸出與 UI 對上、保存開關明顯、可刪除／回放。不設「必須認對」來掩蓋模型限制。

所有 repo tests 僅 synthetic；真人資料與詳細研究產物不進 Git。CI 不能替代 macOS 相機權限／裝置 smoke；無真機 evidence 必須標未驗，不宣稱可用。

## 10. 開工與後續關卡

1. 本 docs-only 設計＋ADR＋母規格／狀態同步經正常 review/merge。
2. 作者依此設計另寫 implementation plan：先 time-boxed capture/UI/加密保存 spike，再 interfaces、session engine、UI、recorder/replay、端到端 evidence。spike 不下載新模型、不先錄真人；其失敗可縮減工具選型，不能放寬隱私或偷偷開 mobile。
3. Operator 核准 plan 並明確下達實作 go 後，lead 才能派產品程式任務；docs merge 不等於這個 go。
4. 研究使用需先完成記錄／刪除驗收和個別同意。研究結果可支持後續演算法比較，不自動關閉 selection gate。
5. Android/iOS、認證、正式部署各有獨立入口 gate；cross-runtime conformance／real-face carrier 在 mobile 評估前仍必須完成。研究影像不自動成為可再散布的跨平台 fixture。

## 11. 未證明假設與停止條件

假設：品質引導與多幀規則能改善 session 結果；Mac 的相機／UI／ORT 在工程預算內；回放資料足以重現策略差異。由上述 spike、真機 smoke、holdout 比較取得證據，不寫成既有效果。

若人物切換不能可靠重置、資料未經同意落盤、無法正常停用相機、原型會改正式 gallery、或 profile 跨版本無法重播，停止真人研究／交付並修正；不得用提高誤接受容忍度或停用 integrity gate 完成演示。
