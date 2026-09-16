# Phase 2B：Mac 辨識研究與有關卡改善閉環

- 日期：2026-09-16
- 狀態：G0 文件與 G1 研究工具工程已批准（decision `d-20260916012440460338-3`）；G3 採集、G4 改善、G5 holdout 未開。本文件描述的研究能力尚未實作。
- 上位規格：[Portable Face Core Design](2026-09-09-portable-face-core-design.md)；研究邊界：[Mac 動態辨識研究設計](2026-09-14-mac-live-identification-research-design.md) §5–§8
- 階段決策：[ADR 0008](../decisions/0008-mac-live-identification-research.md)、[ADR 0009](../decisions/0009-preprocessing-aspect-invariance.md)、[ADR 0010](../decisions/0010-phase2b-evidence-isolation.md)
- 執行計畫：[Phase 2B 執行計畫](../plans/2026-09-16-phase2b-mac-recognition-execution-plan.md)

## 1. 要解決的問題

在指定 Mac、既有固定 one-shot gallery、受控單人使用情境下，系統必須能回答：

1. 本人實際有多少次在有界 session 內被正確認出？多少次認錯其他人、被拒絕或無法完成？
2. 未註冊者有多少次被錯誤接受？不能只報本人成功率。
3. 失敗最早在哪一層出現：裝置／取樣、品質、幾何、身份排序、接受門檻，還是時間聚合？
4. 哪個受控變因實驗支持或反駁這個解釋？改善是否在未參與選方法的新 session 上仍成立？

「已產出報表」不是「辨識問題已解決」；「有 plausible 原因」不是「已證明 root cause」；「development 變好」不是「holdout 通過」。2A 工具結案不因 2B 揭露殘餘失敗而被抹去，但新發現的工具缺陷必須修正後才能支持受影響的研究結論。

## 2. 範圍與非目標

保留：offline Mac、open-set 1:N、每身份一張註冊照、固定 gallery、bounded session、品質最佳單幀與時間一致性兩臂、獨立研究 recorder/evaluator。辨識仍用 `matched`，非 `authenticated`。

本規格增加：可對帳的嘗試分母、足以定位失敗的研究診斷、事先固定的分析觸發、visit 級 development／sealed holdout、配對比較與逐輪決策。

不默許：降門檻、換模型、添加正式模板、online learning、promotion、重註冊作為掩蓋原始失敗的捷徑、手機平台、部署、認證、考勤、500 人準確性主張、大型圖庫收集或雲端辨識。這些都不是診斷階段的自動 fallback。

pyside6 真窗及方形引導仍為既定方向；未經獨立工程與真機驗證，不把 headless bindings 當作 GUI 已完成。是否納入首輪工程包由執行計畫明列，不更換 toolkit 或取消採集形狀契約。

## 3. 四種獨立批准

- **文件批准**：規格與計畫的內容接受；不啟動執行。
- **研究工具工程 go**：只准批准的 synthetic／camera-free 工程；真機 smoke 另需當場確認。
- **採集 go＋個別同意**：指定參與者、目的、資料種類、保存位置／TTL、刪除與撤回；相機由 operator 在場操作。
- **改善 go**：先交證據與最小變更提案，再批准具體修正／校正。一次 go 只覆蓋列明的變因與驗證，不覆蓋任意搜尋。

執行者發現 authority 不明、不同 dispatcher 指令互斥或資料使用目的超出同意，停止相關步驟並回問，不自行推斷授權。

## 4. 研究單位與兩條對照臂

### 4.1 固定共同條件

每個實驗版本固定：程式完整 SHA、環境／execution provider、模型 hashes／generation、alignment contract、one-shot gallery 及 digest、研究 profile、相機選擇證據、取樣／時間語義、採集方式、品質規則、兩臂規則與 tie-break。

起始有界預算沿既有研究設計：5 秒、至多 5 fps、25 張、latest-only queue 1；3 個至少相隔 200ms 的同身份支持是候選臂起始規則。這些不是已驗證的準確率水位。match／review／margin 值必須由具版本來源的 profile 顯式凍結；不從舊 M5 表事後挑格，也不由執行者憑記憶填值。

### 4.2 兩臂

- **A：品質最佳單幀。** 從同一個合格固定窗口依品質選幀，不讀身份分數或標籤選幀；tie-break 可重現。session 結束才可得結果。
- **B：時間一致性。** 同身份、間隔、score、margin、reset／continuity、deadline 都依凍結 profile。首次 terminal 鎖定；不把全窗口最大分數或多數投票偷換成現行策略。

兩臂使用同一封存窗口與同一模型／gallery／品質／score／margin。A 的等待成本包含完整窗口；B 的首次決定時間另報。B 在完整輸入上的首次 terminal 不能被後段更好的幀覆寫。對照是「相同資料與政策下兩種選證據方法」，不是每臂各挑有利的影片。

正常提前停止的短流只能作短流配對或即時診斷，不能冒充完整 5 秒公平對照。cancel／多人／error 仍立即停止，不為湊滿窗口繼續錄。

### 4.3 三種 replay 不混名

1. **決策重演**：保留原 observation 順序及原 captured／processed／deadline 相對時間，驗 terminal、support、reset 是否重現。
2. **同幀離線重推論**：用相同影像重跑 detector／embedder，研究分數／幾何差異；其處理速度不是原現場 latency。
3. **反事實實驗**：只在 development 及明確批准的變因下改 profile／程式；不得覆蓋原 run 或算成新真人嘗試。

重新排序、重編 sequence、補造漏失幀、用 replay 的快時鐘取代原處理時間都可能改變 B 的答案，必須被檢出而非自動「修好」。

## 5. 分母與結果契約

在已具記錄同意且接受 Start 的時點建立 encrypted attempt 記錄，**先於 camera open／scoring**。每次 Start 一個 attempt ID，重跑 replay 不新增真人 attempt；重試保留 parent attempt，不能用成功那一次替換前次失敗。

- **attempted**：接受 Start 的研究嘗試，包含其後的 open failure、cancel、timeout、error、無影像、無標籤。
- **labeled**：有有效且版本化 ground truth 的 attempted；值為 enrolled identity／unenrolled／uncertain。uncertain 是標記狀態，不能算已知真值；監督指標另列 truth-known 數。
- **eligible**：針對每個指標明列 eligible set，而不是單一「好看的 session」集合。無臉／低品質／timeout 是操作失敗，**不能從端到端分母消失**。
- **paired-replay eligible**：兩臂具有相同可驗證完整窗口、版本相容且無缺檔／tamper 的集合。這是策略比較分母，不是全部真人辨識成功率分母。
- **successful**：enrolled truth-known session 的 terminal=`matched` 且 identity=truth；非單純 status=`matched`。

每臂分開報：correct enrolled match、wrong enrolled match、unenrolled false accept、review、unknown、timeout、invalid_input、error、cancelled、unlabeled／uncertain、refused replay。終局分布與真值交叉分類相互對帳，但不強迫兩者用同一分類軸。

最少展示：enrolled correct／所有 enrolled truth-known attempts；enrolled wrong／同分母；unenrolled false accept／所有 unenrolled truth-known attempts；以及相同指標在 decision-capable／paired eligible 集合的條件版本。分母為 0 顯示 `not estimable`，不可顯示 0%。

sampled/scored/staged/replayed/support 幀數分開，不作獨立人次。time-to-correct、time-to-wrong、nondecision 到 timeout／cancel 的時間分開；失敗不以成功 latency 排除，不把截止時間當作已觀察的認出時間。

## 6. 最小診斷資料與 truth 隔離

研究 trace 必須回答「每個 observation 為何被採用／拒絕、窗口何時清空、為何未成功」。資料留在已同意、加密、repo 外的 research store；一般 logs／終端／PR 不含逐人可識別資料、pixels、embeddings、原始個人路徑。

最小診斷層級：

| 層級 | 必要證據 |
|---|---|
| provenance | attempt／visit／split、code/profile/model/gallery 版本、同意與到期時間、採集方式、裝置解析與實際 shape |
| capture | 原 sequence／captured time、sample/stage 對映、取樣與 dropped 計數、window 完整性、缺失原因 |
| pipeline | detector face count／confidence、face box／landmarks、品質各項值及拒絕原因；幾何契約與失敗原因 |
| scoring | 合格且實際進入 scoring 的每身份分數或可無損重算的已同意輸入；top1/top2、margin及距門檻差；未 scoring 要明示 missing reason，不以0代替 |
| session | processed time、reset reason、當前連續支持身份／序列、支持間隔、terminal發生點與deadline、baseline選幀依據 |
| evaluator only | truth identity的rank/score、最高非truth score、truth gap、錯誤類別、label版本及分析結果 |

truth gap 定義為 `score(truth) − max(score(other identities))`；它不同於 top1−top2 margin。unknown 沒有 truth score/rank。truth 只由 evaluator 在決定鎖定後 join；不得傳進 scorer、quality rank、採集提示、支持規則或選幀。

不為填滿 trace 繞過品質 gate 去 embed 原本拒絕的幀；先用 face/quality/geometry 證據分析。確有必要的 bypass 反事實另提受控實驗批准，永不作產品路徑或正常成功指標。

不常態保存衍生 crop／tensor／embedding。若要檢查幾何，優先從仍在 consent／TTL 內的加密幀在本機記憶體重算；任何新增持久資料種類需先完成規格、同意、TTL與刪除鏈驗收。影像7天／評估30天沿既有預設；不能靠複製衍生資料延長影像用途。未實作的診斷欄位必須由工程 gate 補齊，不能假定現有 envelope 已有。

## 7. 取得數據即觸發的分析規則

每個小批次在下一批採集前必須完成：分母對帳、replay validity、錯誤分層、trigger registry、下一步決策。不存在「先收完很多資料再看看」。

| Trigger | 必須做的事 | 禁止推論 |
|---|---|---|
| 任一 wrong enrolled matched／unenrolled false accept | 立即隔離候選改善的放行；保留合規證據、建 incident；核 truth/provenance、逐幀支持來源、最高冒名身份與錯誤首次出現層 | 不能用 overall accuracy 或3幀一致掩蓋，也不自動判模型壞 |
| 任一 enrolled 未正確認出 | 每次列出最早阻斷層；同 signature 重複或阻斷全部普通情境時，升級有界 RCA 實驗 | timeout 不是一律「分數太低」 |
| 任一 live／decision replay terminal、identity或support不一致 | 阻擋該資料的比較結論，先查seq、時鐘、完整窗口、版本及stage mapping | 不能直接稱模型非決定性 |
| 任一分母／label／split／missing-file不一致 | 停止統計主張，修 evidence plumbing，保持原 attempted 不消失 | 不能刪失敗列讓兩臂變配對 |
| 高品質幀反覆 wrong rank或truth gap≤0 | gallery provenance／one-shot覆蓋／alignment／模型表徵依序提出控制；檢查錯人是否穩定 | 降接受門檻不能改rank |
| truth rank=1但未通過score或margin | 分開數出score-only、margin-only、both、None runner-up；展示與unknown分布重疊 | 不能因本人差一點就選新門檻 |
| A正確、B未成功 | 查有無3個間隔足夠且在deadline前可用的連續支持、reset及identity交替；比較原時序重演 | baseline診斷matched不等於B terminal matched |
| B錯誤且支持穩定 | 判為相關的系統性錯誤，回到排序／gallery／模型層 | 多幀不是獨立身份證據 |
| 低有效幀／大量drop／late processing | 分開capture、quality、compute bottleneck；估計可用支持是否根本不足 | 不能先延長session或取消quality gate |
| 同controlled condition重複惡化 | 做分層與單變因配對控制，再決定是否涉及capture guidance或表示能力 | 關聯不是root cause |

安全違反（不同意落盤、外傳、gallery mutation、key/integrity失敗、無法停相機）立即停止真人研究；辨識錯誤本身可以是已同意研究的觀察，但不准繼續無限重試、擴充採集或作有效性宣稱。已預先批准的小批次是否可繼續由 lead 依 incident 分類記錄；涉及新採集目的則回 operator。

RCA 必須交：觀察／假說／至少一個競爭解釋／控制組／唯一改變因子／凍結項／可反駁預測／原始evidence locator／結果／結論強度／下一步。不接受只有「可能是光線、模型或照片」的清單。

## 8. Development 與 sealed holdout

依 participant／visit／session 做切分；相同visit、連拍、裁切或重播不可橫跨development與holdout。新subject／新visit／新frame是不同的泛化問題；同一人未來visit只能支持該人時間情境下的結果。

holdout 必須在觀察辨識結果前指定；分析方在候選凍結前不可取得其影像、標籤、分數、結果或據此的調整建議。只許不暴露內容的完整性／數量檢查。角色隔離不足時，採「候選凍結後才採未來visit」的 prospective holdout，不能把同一分析者已看過的資料換資料夾就稱 sealed。

解封前凍結：候選full SHA/profile、唯一primary comparison、指標、排除規則、樣本與停止預算、允許的主張。只評一次；看過結果再改策略即成下一輪development，重新取得未使用的新資料才可再次作holdout。

小樣本只作描述／可行性證據；同visit相鄰session高度相關，不以幀數或session數假裝獨立參與者。0/N false accept 只表示本批未觀察到，不等於真實風險為0。若需要可量化風險上限或非劣性證明，另由operator決定風險容忍與獨立樣本設計，不能由模型自行指定安全水位。

## 9. 改善回路

1. development 基線及分層證據完整。
2. 一次只選一個有證據的失敗signature，先便宜且能反駁的控制，不列模型／門檻無限搜尋清單。
3. 若找到工程缺陷，另立最小修正與RED→GREEN；若只是策略／模型局限，先提出比較方案，不稱bug已證實。
4. 提案列變更位置／因子、控制、預期改善的指標、誤接受保護、rollback、驗證資料與所需同意；operator go之後才執行。
5. 原baseline不改，candidate新版本；重新採集或相同development的配對只算development evidence。
6. 候選凍結後用sealed／prospective holdout核驗。未通過就回報失敗／資訊不足，不換metric、不調門檻再重考同一份試卷。

## 10. 完成狀態

- `TOOLING_READY`：量測、回放、分母、診斷與刪除證據成立，不宣稱辨識有效。
- `BASELINE_CHARACTERIZED`：事前約定的小批次完成，錯誤可定位到層；未註冊者不足必明列。
- `CAUSE_SUPPORTED`：有受控實驗支持，清楚限定樣本／情境；未排除的競爭解釋仍列出。
- `IMPROVEMENT_SUPPORTED_LIMITED`：凍結candidate在未用於選方法的新資料上改善預定指標，且未觀察到安全退步；僅限所測人員／裝置／情境，非deployment acceptance。
- `NO_IMPROVEMENT`／`INCONCLUSIVE`／`BLOCKED`：合法的研究終局，但不得宣稱辨識問題已解決。

2B 研究交付完成與「operator認為Mac辨識已足夠可用」分開。後者須operator基於證據明確判斷；本規格不預先保證任何準確率，也不替operator接受誤認風險。

## 11. 規格落地 acceptance

1. 明列四種批准、固定one-shot/open-set、兩臂公平窗口、truth隔離。
2. 分母含所有接受Start的失敗；逐臂結果／refusal能對帳；成功不是僅status=matched。
3. 採集前確認診斷欄位與刪除鏈可用；每批必執行trigger table並交RCA／下一步。
4. visit切分、sealed／prospective流程、污染處理與小樣本主張限制明確。
5. 改善只能在證據＋operator go後；新資料驗證與原baseline永遠分開。
6. 新文件掛入map，PROJECT-STATE僅寫當下授權與短pointer；不得把計畫中的能力寫成已實作。
7. 無照片／embedding／DB／權重／逐人詳細trace進Git或雲端；舊M1/M3/M5不改寫，不以其資料選2B成功門檻。

## 附錄 A：採集幾何契約

本附錄定義 2B 研究採集的影像幾何契約。它是功能面權威；執行計畫對同一主題的敘述若與此不符，以本附錄為準。

### A.1 適用範圍

適用於 2B 研究採集路徑取得的每一個 sample frame，含 Qt 引導窗與 CLI／synthetic 演練。不適用於 2A 既有產品路徑。

### A.2 方形裁切定義

原 sensor frame 先經 orientation 規範，再依下式取中心方形：

- 邊長 `S = min(W, H)`
- 左上原點 `((W - S) // 2, (H - S) // 2)`

裁切不得伴隨非等比縮放。此裁切是研究採集的形狀契約，不因裝置回報的 shape 差異而放寬。

### A.3 Overlay 與實際 crop 使用同一 mapping

畫面上顯示的方形引導框與實際送入 inference 的 crop，必須由同一組 mapping 計算。不得一邊用顯示座標、另一邊用 sensor 座標各自推導。

引導框與實際 crop 不一致，即為契約違反，而非顯示誤差。

### A.4 Mirror 只影響 preview

鏡像僅作用於使用者所見的 preview。它不得改變：

- 實際 crop 的座標
- 送入 inference 的像素
- 保存的 sample frame
- 記錄的 `crop_mapping`

### A.5 下游 pipeline 不因 capture square 而改變

既有的 640 letterbox 與 112 align 不因採集端已為方形而取消或跳過。capture 端的方形契約與 pipeline 端的尺寸規範是兩層，不互相替代。

### A.6 identity 不參與 crop

crop 的位置與大小僅由幾何（`W`、`H`）決定。任何身份分數、標籤、辨識結果或 gallery 內容不得進入 crop 的計算。這是 truth 隔離在採集層的延伸（見 §6）。

### A.7 保存策略（第一輪固定）

第一輪固定採用：沿既有有限全幀保存，另記 `crop_mapping`；不額外保存第二份 crop。

已同意的原 sample frame 與所記的 `crop_mapping`，必須足以重建出同一份 inference 輸入。

若後續改為只保存 square input，則必須在 manifest 明示不能回看框外，不得冒稱保留了 sensor 全幀。此變更需先完成規格、同意、TTL 與刪除鏈驗收（見 §6 對新增持久資料種類的要求），不得於實作時逕行切換。

### A.8 不得宣稱的事項

完整背景仍受單人受控場地與 consent 約束。不得宣稱 crop 框外的人必然會被偵測，也不得以本契約作為框外涵蓋範圍的保證。

### A.9 未定項

以下由 plan 與本附錄留待後續凍結，實作時不得自行決定：

1. orientation 規範的具體來源（裝置回報值、EXIF、或固定假設）——本輪採 fixed synthetic assumption；真機階段再議。
2. `W`、`H` 為奇數且 `W == H` 時的退化情形處理——依 A.2 公式得 `S = W = H`、原點 `(0, 0)`，行為確定；任一邊為 0 時 fail-closed，不補值。
3. 本輪已由 E7-B premise 凍結 `crop_mapping` schema：`{"x","y","size","frame_w","frame_h","mirrored_preview"}`；後續變更需另行規格、同意、TTL 與刪除鏈驗收。
