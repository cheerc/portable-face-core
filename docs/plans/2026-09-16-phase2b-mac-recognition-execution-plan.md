# Phase 2B Mac 辨識研究執行計畫

**版本：** v1.1，2026-09-16。**狀態：G0（文件）與 G1（研究工具工程）已由 operator 於 2026-09-16 批准，記於 decision `d-20260916012440460338-3`。G3 採集、G4 改善／調參、G5 holdout 解封仍未開。**

**Source of truth：** 母規格 `docs/specs/2026-09-09-portable-face-core-design.md`；Mac 研究設計 `docs/specs/2026-09-14-mac-live-identification-research-design.md` §5–§8；ADR 0008／0009／0010；配套規格 `docs/specs/2026-09-16-phase2b-mac-recognition-research.md`（與本計畫同批落地）；方向批准 `d-20260916004436411112-2`；G0／G1 批准 `d-20260916012440460338-3`。

**基線：** `cheerc/portable-face-core` @ `dab9220c7e1241afddb36679e2b1642e84c73adc`。執行前由 lead 核新 HEAD 的受影響面，不強迫 checkout 過期版本。

**Goal：** 在 operator 的 Mac 上，量出真實 session 的正確認出／錯認／拒識與延遲，將每類失敗連到可反駁的原因實驗；經另行批准的最小改善，要能用未參與選方法的新 session 檢驗，而不是只讓同一批資料分數變好。

**Architecture：** 延用 capture→單幀 pipeline→純 session engine，以及獨立加密 research recorder/evaluator；補足研究 evidence，不建立第二套辨識 policy。ground truth、split 和診斷因果分析留在 evaluator 邊界，不能回流 scorer 或選幀。

**Tech stack：** 既有 Python／NumPy／ONNX Runtime、pytest／ruff／mypy 與 research CLI；既定 GUI 方向 pyside6。具體 dependency/命令以 source inventory 及執行時 lockfile 為準，不安裝或下載新模型作為本次寫計畫的副作用。

**文件性質：** operator 2026-09-16 指示本計畫與配套規格一併進 Git，覆寫 commander 預設的「plan 只落自身 workspace」慣例；本檔落 `docs/plans/`。它仍是**拋棄式執行計畫**：執行完不維護、不回填，功能契約永遠以 spec 樹為權威，兩者衝突時以 spec 為準。舊 `2026-09-14-mac-live-identification-implementation-plan.md` 是已完成的 2A 歷史計畫，不在其上追加 2B。

## 0. 接手者先讀這一頁

### 0.1 你現在可以做什麼

**已開：** G0 文件落地、G1 研究工具工程（§12 的 E1–E8，camera-free／synthetic）。每個執行 task 仍由 lead 建立唯一 board identity、assignee、dependency 與完整 brief 後才開工；本文件的 E 編號不是已派出的 task，R／H 編號更不是——它們屬於尚未開啟的 G3／G5。

**仍未開：** G3 真人採集與個別同意、G4 改善／調參／換模型實作、G5 holdout 解封。

未取得對應批准前，不做：開相機（E8 的真機 smoke 需 operator 在場另外安排）、讀真人圖片內容／跑真人推論、保存新的真人資料、調參、換模型、換註冊照、改正式 gallery、learning／promotion。工程期間一切驗證用 synthetic 或 test doubles。

### 0.2 讀後必須能回答的三件事

1. **現在卡在哪一層？** 不能把 `timeout` 一概寫成「模型辨識不好」。
2. **哪個實驗能排除另一個解釋？** 不能只多收一些資料、換一串參數。
3. **這次宣稱靠的是 development 還是未碰過的新資料？** 不能用同一份樣本選方法又宣布改善已驗證。

### 0.3 最短路徑

文件／同意邊界凍結 → 最小 evidence 工程 → synthetic 端到端與刪除驗收 → operator 在場非真人相機／GUI smoke → 小批 development → 每批分析 → 一個有證據的最小改善提案 → operator go → candidate development 驗證 → candidate 凍結 → prospective holdout → 有界結論。

沒有改進證據時，合法出口是 `NO_IMPROVEMENT` 或 `INCONCLUSIVE`；不保證一定找到可用模型，不把輸出報表當成「辨識度問題已解決」。

## 1. Global Constraints

1. offline、open-set 1:N、固定 one-shot gallery；每身份一張既有註冊照，gallery 至少兩身份才有 runner-up。不移除「容易混淆的人」美化 margin。
2. baseline 的 code／model／gallery／profile／採集方式均凍結。candidate 只改明確批准的一個因子；跨 generation 重建 gallery，不混 embedding。
3. baseline A 按品質選幀，B 依時間一致性；不換成最高身份分數、any-frame-wins、未批准 score fusion 或 adaptive。
4. 5 秒／至多5fps／25幀／queue1 是現有研究預算，不保證實際達5fps；慢也不能延長 deadline。任何更動要新版本與 operator go。
5. 研究同意、影像保存同意、learning confirmation 分開。第一輪無 learning/promotion，也不修改正式身份庫。
6. 真人照片／embeddings／DB／權重／逐人trace不進Git、一般log、Artifact或雲端；敏感診斷只在本機加密store。禁止plaintext temp、螢幕截圖旁路、未知備份／同步。
7. 影像TTL7天、評估TTL30天沿既有研究預設，須採集前再次確認；衍生資料不能繞TTL。缺key／tamper／失敗刪除不降級明文。
8. `matched` 非 `authenticated`；多幀不是liveness或獨立身份證據。mobile／部署／考勤／500人泛化不在本計畫。
9. 舊M1/M3/M5只作歷史入口與方法限制；不得改舊報表、用舊threshold sweep選本次門檻、把v3彙總補寫成不存在的逐筆log。
10. 任何工具問題先保留evidence並由lead走bug入口；不由commander改碼、不繞integrity/generation/worktree gate。

## 2. 起點：已知事實與未證明假說

### 已知

- PROJECT-STATE 已宣告2A結案，工具鏈與contract v3前處理修復已落地；這不等於辨識正確性通過。
- 本輪核得遠端main為上述完整SHA、open PR為0、該SHA CI success。canonical本機HEAD落後，不能用其檔案冒充遠端現況。
- PROJECT-STATE明列沒有保留的真人session。過往smoke只能證明當時鏈路與指定驗收，不供本輪雙臂公平重播。
- 研究設計要求品質最佳單幀對照、時間一致性、session分母、visit切分及未註冊者；舊runbook／plan中「待完成」敘述需依現行碼證區分。
- 原研究spec已要求pyside6真窗與方形採集；PROJECT-STATE仍列真窗未實作。不能把headless事件bindings當作真窗。
- research runbook §9c明確分開scored與staged，不應用它們互相作分母。

### 假說，不先當結論

| 假說 | 怎麼判斷 | 責任人／出口 |
|---|---|---|
| 多幀會比品質單幀更好 | 同一固定窗口、相同門檻、配對結果與time-to-decision | lead組織R批次；可反駁，不承諾B勝 |
| 主要殘差來自單張註冊照覆蓋 | provenance先過；development中truth-rank/gap與條件對照，必要時另批隔離註冊照診斷 | 分析owner提實驗；不直接換正式照 |
| 只要降低margin就能認對 | rank與score/margin分開；unknown及錯人分布對照 | 不在baseline動門檻；校正另請go |
| pipeline修好就不會再有幾何問題 | 既有conformance＋同幀真路徑diagnostic evidence | 發現反例另立最小bug，不重開全部2A |
| 現有replay能重現live decision | 原sequence／timestamp／processing／terminal replay一致性 | E系列驗證；不一致先修evidence |
| 目前欄位已足夠定位原因 | source inventory＋synthetic故障逐項注入 | 缺欄位先補，不開始正式採集 |

## 3. 授權、文件與停止關卡

| Gate | 放行前必有 | 不通過時 |
|---|---|---|
| G0 文件 | 新2B spec、計畫、診斷隱私邊界經operator審閱；lead完成docs-check及正常文件流程 | 只做已授權唯讀盤點；不執行工程 |
| G1 工程 | operator明確研究工具工程go；每task有source、assignee、scope、RED/GREEN及review class | 不建立實作派工 |
| G2 工具 | E系列synthetic acceptance、fault matrix、真入口鏈、stop/delete/restart evidence齊 | 不採真人，回最小缺口 |
| G3 採集 | operator在場；每參與者個別同意；store/key/TTL/profile/裝置/label/split/sample budget固定 | 未知寫BLOCKED，不借用舊同意補空格 |
| G4 改善 | development RCA＋最小變因提案、風險與rollback、operator對該變更go | 只報原因證據，不試調參／換模型 |
| G5 holdout | candidate與primary comparison凍結；未使用資料、事前樣本／排除規則、custodian明確 | 不解封，不看分數選candidate |
| G6 結論 | 逐臂分母、配對差異、安全事件、覆蓋限制與原始evidence可核 | 不能宣稱改善成立或部署可用 |

**docs-check建議：arch=N adr=Y area=Y。** 保持既有核心架構；新增研究attempt/trace生命週期與split隔離是資料模型／跨模組邊界決定，須在實作前落文件。搜尋證據來自母規格§6/12/14、研究spec§5/6/8、ADR0008、PROJECT-STATE、runbook§9b–d；source inventory再核完整docs map。

**D0為所有E工作的blocking dependency：** lead執行`project-docs-maintain`，新增2B spec，更新PROJECT-STATE／README短pointer，並以ADR記下「truth／split不入inference、敏感trace沿同意與刪除鏈、raw資料不進公開報表」取捨。建議ADR檔名`docs/decisions/0010-phase2b-evidence-isolation.md`，落地前確認編號空缺；若被占用由lead選下一號並同步pointer。原2A plan不改寫，原研究spec只加新spec pointer，不完整複製新規則。

**立即停止真人研究：** 未同意落盤、外傳、正式gallery被寫、完整性／key失敗、相機／worker無法停、label進入policy、holdout洩漏。辨識錯誤則立即建incident與阻擋candidate放行；可以作研究觀察，但不能無限重試到成功。

## 4. 採集前凍結的 Experiment Manifest

以下是**擬新增研究契約**，不是宣稱現有CLI已支援的keys。G3不得留下空值；缺一項由lead退回確認，不以預設猜測。

| 區塊 | 必填內容 |
|---|---|
| identity | experiment_id／schema_version、批准decision／task refs、owner／custodian角色 |
| software | code full SHA、Python/OS/arch、實際ORT/provider與依賴版本、model manifest/checksums、align contract/generation |
| gallery | 原始one-shot corpus manifest digest、每身份模板數=1、gallery digest、身份數、合法使用與同意ref（敏感映射另存） |
| policy | 完整profile及hash、match/review/margin/detector/quality規則、support/interval/continuity/deadline、tie-break |
| capture | request device及實際選到裝置證據、shape/color/orientation/mirror、採集模式、方形crop規則及座標映射版本 |
| privacy | 非Git／非同步store、key namespace、records/images同意、TTL、刪除／撤回操作、diagnostic field allowlist |
| study | participant opaque IDs、enrolled/unknown型別、visit grouping、情境、次序、planned attempts、retry policy、dev/holdout assignment |
| analysis | A/B定義、primary outcome、trigger版本、batch-stop、paired exclusion、timing語義、允許主張 |

**profile來源規則：** 先確認現行external profile的來源、完整內容與合法性；如果不存在可追溯的profile，G3 BLOCKED，交operator凍結起始profile。不能拿M1的match、M3的review、M5的margin拼一套，也不能用第一次採集結果倒推起始值。

同一batch中gallery/profile/model變化即停止該batch，保留已完成嘗試；新manifest產生新batch，不覆寫舊值。

## 5. 研究順序與採集預算（提案，G3批准後才執行）

### R0：零真人演練

用synthetic序列走真入口，蓄意製造wrong-rank、score-only block、margin-only block、reset、missing blob、late frame、open failure、cancel及unknown FA；輸出的trigger與case分類必須符合§8，不准以「tests綠」代替檢查整份報告能否回答研究問題。

### R1：第一個development小批次

最小可行是**一位已註冊參與者＋一位同意的未註冊參與者**，但gallery仍保留原完整固定集合，不縮成兩人。這只支持這些參與者，不是整個gallery每個人都通過。

每位每visit：普通正面、較低但安全可見的室內照明、輕度轉頭三情境；每情境預先固定2次attempt，共6次。三情境的具體站位／照明／轉頭指令先用文字固定，不宣稱有未裝設的lux meter或精準角度量測；實際quality值另記。順序於採集前固定或以固定seed分配，不能看結果再挑順序。

這是**小型診斷覆蓋預算，不是統計power或準確率驗收門檻**。需要配件／特定姿態的情境只有在operator認為重要且參與者同意時另版加入，不為填表索取資料。

- 第一visit：最多12個attempt（兩位各6）。
- 第一visit分析完成後，才決定第二development visit是否有必要；同意與目的仍有效才可執行。第二visit需重新進場／重新定位，不把連續錄影切段冒充獨立visit。
- 初始development最多2個visits，即最多24個attempt；不是「做滿才准分析」。每一visit後都停下做R2。
- 未註冊者未到：只准已明確批准的enrolled開發診斷；unknown欄填`untested`，禁止過G5／宣稱open-set改善完成。
- 參與者中止、無法重現情境或隱私問題立即停止；保留attempted、原因及撤回後允許留下的非生物audit，不補造資料。

### R2：每批強制分析，不等收完

固定順序：版本／完整性 → attempt對帳 → label合法性 →雙臂 replay一致性 → outcome交叉表 → 最早阻斷層 → §8 trigger → case packet → 下一步。

每批只允許四種下一步：`繼續原已批准採集`／`先補工具證據`／`提出一個受控RCA`／`停止並回operator`。下一批採集不得靠「還想多看看」續行。

### R3：有界RCA

優先一個signature、一個主假說、一個競爭假說。每個RCA最多先規劃兩個單變因控制；每個控制一份完整evidence packet。半個工作日為首次分析time-box（不含等待operator或參與者），到時仍無區分力就報`INCONCLUSIVE`＋缺哪個觀察，不擴成無限模型搜尋。

**先重用development中仍合法存活的加密資料，後考慮新採集。** TTL到期就停止使用；不能把資料複製到新store延命。

### R4：改善提案與另行go

提交：問題signature、原始證據、控制結果、最小變更、影響caller、保持不變的條件、預期幫助哪個metric、unknown／wrong-id保護、rollback、development測試與新資料驗證方案。若有兩個彼此獨立的根因，拆兩單；本輪最多先批准／實作一個candidate，不一起改alignment、threshold與gallery後只交總分。

candidate在development表現不佳不進holdout；最多兩輪有明確新證據的candidate提案，超過回operator重定範圍。這是工時／資料止損，不是實作預先授權。

### H1：建議使用prospective holdout

首選在candidate、評估程式與manifest凍結後，才請相同參與者安排**未來的新visit**，以相同三情境各2次（兩位最多12個attempt）檢驗baseline/candidate兩版本；兩版本在每個相同新bundle上配對。這樣不需要pretend同一位分析模型能忘記已看過的影像。

若改善的因子是採集引導而非離線policy，無法對同影像重播兩種採集方式；必須改成事前排定的A/B或交錯次序實際採集，標為matched-condition而非same-input causal comparison，operator另批預算及同意。不得沿用上述同bundle配對名義。

預存sealed holdout只有在有獨立custodian、分析者拿不到內容／分數／標籤、存取可稽核且TTL足夠時才用。解封後不看結果再選candidate；若改策略，這批轉development，新一輪需新資料。

## 6. 指標與對帳：先算對，再談改善

### 6.1 兩套軸不要混

`status`：matched/review/unknown/timeout/invalid_input/error/cancelled，依各臂實際終局。

`evaluation class`：correct enrolled match／wrong enrolled match／unenrolled false accept／enrolled nonmatch／unenrolled nonaccept／truth unavailable。

例如 `status=matched, identity≠truth` 是wrong enrolled match，不是successful；`status=timeout, best_baseline=matched` 仍是B的timeout。

### 6.2 必交分母表

- 全部接受Start的attempted；有效標記labeled；truth-known enrolled／unknown；uncertain／unlabeled。
- 每一arm的terminal及無結果refusal；各metric eligible／ineligible原因。
- E2E分母包括camera error、quality failure、timeout、cancel，不以paired replay eligibility排除這些。
- `paired_complete`只用於共同完整窗口的A/B或baseline/candidate比較；不足者仍在attempted與failure表。
- 同一attempt replay十次仍只有一個真人attempt；一個visit六次不是六位獨立參與者。

主要率：`correct_enrolled / known_enrolled_attempts`；`wrong_enrolled / known_enrolled_attempts`；`unknown_false_accept / known_unknown_attempts`。同時列count與denominator；分母0=`not estimable`。

同資料的策略差異用配對表：A對B對、A對B錯/未決、A錯/未決B對、兩者都錯/未決；wrong-match與nonmatch不可為了配對而隱藏合併。baseline/candidate另有對應表與逐條case refs。

### 6.3 時間

分開capture時限、原始processed延遲、首次terminal、time-to-correct、time-to-wrong及無決定截止；成功latency不是全體latency。對沒有correct事件者保留right-censored觀察，列取消／錯認／error競爭終局，不把5秒填成認出時間。

A按全窗口挑幀，等待成本到窗口末；B的first terminal可較早。offline重推論耗時另表，不替換原live耗時。

### 6.4 小樣本

首輪不作人口準確率／低誤認保證／統計顯著宣稱。不將幀作n，不用小量cluster bootstrap產生虛假精度；報participant、visit與attempt三個數。若將來operator要求風險上限或非劣性，需要獨立樣本設計與事前容忍值，另立statistical plan。

**改善有限成立的最小判準：** 在事前固定的holdout配對集合，candidate的correct enrolled count高於baseline、wrong enrolled及unknown FA皆不高於baseline、無新hard-error/privacy回歸；逐人／情境及時間代價完整揭露。這只支持`IMPROVEMENT_SUPPORTED_LIMITED`，不證明安全非劣性。如果baseline已全對，另批主要目標（例如時間）才能比較，不能事後換metric宣布勝利。任何wrong/FA即使兩者相同仍需incident，不能稱安全可部署。

## 7. 診斷計算與因果強度

### 7.1 最早阻斷層

對enrolled未correct的attempt，按**第一個無法滿足的必要條件**分類：

1. start/open/capture失敗或provenance無效。
2. 沒有足夠進入pipeline的時序觀察（sampling/drop/late）。
3. detect／single-face／quality不通過。
4. 合格幀有scores但truth從未成為top1（rank失敗）。
5. truth可top1，但score或margin／runner-up不可用阻擋。
6. 有單幀接受truth，但支持數／間隔／reset／deadline阻擋B。
7. B先對錯人terminal，後段truth再好也不能覆寫。
8. evidence不足不能區分：`UNRESOLVED_EVIDENCE`，列缺欄位；不得硬分到模型錯。

次要原因可多標籤，但主因分類必可由trace重算。這是定位假說的操作分類，不直接等於root cause。

### 7.2 evaluator-only衍生量

對真正scored的每幀：truth rank（ties規則明列）、truth score、highest nontruth score、truth gap、top1/top2、top1 margin、`top_score-match_threshold`、`margin-margin_threshold`。quality gate擋下無score就標missing reason，不能填0。

分別計：truth-top1比例、wrong-top1穩定度、top1切換、最長合法支持run、reset原因分布、可用支持間隔、deadline前來得及完成的幀數。以上是同session診斷，不當獨立樣本置信證據。

### 7.3 結論用語

- **觀察**：實際看見了什麼，不說原因。
- **相關**：分層有差異，其他因子仍混雜。
- **機制相符**：時序／內部證據與預測一致，但控制不足。
- **受控支持**：只改一因子、相同development控制、符合事前預測且競爭解釋被削弱。
- **新資料支持**：凍結candidate在未用於選方法的新session仍有預期效果。

不能從「某張註冊照與live不同」跳到「必須加模板」；不能從「brightness低」跳到「光線就是主因」。

## 8. 深入分析觸發表（接手者逐列執行）

| ID／觸發 | 優先核查 | 最小區分實驗 | 必交證據／停止條件 |
|---|---|---|---|
| T01 任一wrong enrolled matched或unknown FA | truth／gallery對應、scope/generation、錯人首支持與terminal | 原observation決策重演；判斷單幀排序已錯或engine組合出錯 | incident、錯誤首次層、A/B對照。阻擋candidate放行；不以提高success抵銷 |
| T02 同inputs/profile重演不一致 | 原seq、stage mapping、captured/processed time、deadline、版本 | 先不用重新infer，原observation重演；再同幀infer對照 | 對位第一個分歧event；該批策略比較HOLD，不能只看最終分數 |
| T03 任一漏attempt／refusal／split錯 | attempt ledger、commit/error/label sidecar、配對集合 | synthetic open error／missing middle frame／one-arm refusal | 修報告前不出準確率；不能刪出事session |
| T04 enrolled有合格幀但truth rank始終>1 | gallery identity映射、註冊pipeline、wrong-id是否固定、geometry | 同一已同意development幀走原入口與正規化入口；geometry contract控制先於換模型 | truth-gap／wrong-top1分布、控制差異。若無幾何反例，不能再斷言形變根因 |
| T05 truth rank=1但無matched | score-only／margin-only／both／None runner-up分開 | 不改門檻先列四格；與unknown同分布比較；若提校正，先G4 | 原profile各distance-to-gate、unknown風險；禁止「剛好過本人」閾值 |
| T06 A正確而B timeout | 最長連續合法支持、identity alternation、quality/reset、elapsed | 原時間重演；approved development反事實只移除一種非安全時間限制作因果診斷（不得入產品或算成功） | 哪個event清空窗口、是否有足夠3幀。需要反事實批准時先停，不直接放寬continuity |
| T07 B穩定認錯 | 支持幀是否同一錯人、共同品質／pose、truth gap | 同一固定gallery對A/B；穩定wrong rank與孤立spike分開 | 明示多幀相關錯誤；禁止把3次支持稱3份獨立身份證據 |
| T08 全quality拒絕或有效幀不足 | face count、face size、sharpness、exposure、pose、reject reason | 在已批准採集內只改一項引導（距離／光線），以新paired-condition session核變化 | quality components與端到端結果同列；禁止先bypass quality/embed rejected幀 |
| T09 late/drop高且拿不到支持 | capture間隔、每階段耗時、processed-deadline差、queue drop | synthetic慢scorer與真capture分開；模型已初始化前後別混 | runtime profile與性能證據；不延長deadline掩蓋慢 |
| T10 某情境／visit反覆差 | 相機選到誰、shape、gallery/profile同一、照明/pose標註 | 相同visit內事先設計單因子控制；新visit作驗證 | 層級差異與confound；不把檔名當時間或跨日證據 |
| T11 每幀score/margin整體塌縮或幾何異常 | orientation/RGB、letterbox反變換、landmark對齊、model contract | face-free conformance＋同一development輸入的representation控制 | 變換參數／容差引用／差異；不能預設contract v3已永遠無bug |
| T12 candidate看似提升但多了eligible/少了error | attempted一致、excluded列表、A/B拒絕、profile/gallery是否暗變 | 全attempt表重新join；同intersection及E2E兩種口徑並列 | 分母守恆與差異原因；只提高條件成功率不算E2E改善 |

T01–T03每件觸發即處理；T04–T12每件先歸類，同signature在兩次attempt重現或一個batch所有普通情境均失敗就啟動R3。兩次是工作分流閾值，不是統計證明；單次嚴重失敗也可由lead提前升級。

## 9. 每個分析case的交接格式

以下全留repo外加密store；Git／對operator摘要只留不含個資的case code與aggregate。

- `case_id / trigger_id / owner / source attempt IDs / evidence expiry`
- **Observation：** 原終局、truth類別、A/B差異、最早阻斷層；不混診斷top1與terminal。
- **Hypothesis H1 / alternative H2：** 各寫一個可被否定的預測。
- **Control：** 相同輸入／同情境哪一種；哪些hash保持不變。
- **Intervention：** 唯一因子、批准ref；不改truth，不接learning。
- **Evidence：** encrypted locator、run IDs、seq範圍、實際命令／版本／exit、原始輸出摘要、失敗和缺資料。
- **Result：** 支持／反駁／不能區分；numerator/denominator及scope。
- **Decision：** 繼續、工具bug、提出改善go、或停止。列下一步owner與gate，不寫「再研究」。

如果raw資料已到期，case保留合法的評估摘要與「不可重播」標記；不能繼續宣稱full reproducibility。

### 9.1 接手模型必須通過的六列算例（純合成，不是真人結果）

| Attempt | Truth | A終局 | B終局 | 資料狀態 |
|---|---|---|---|---|
| s1 | 已註冊p1 | matched p1 | timeout | 完整固定窗口 |
| s2 | 已註冊p1 | matched p2 | matched p2 | 完整固定窗口 |
| s3 | 已註冊p1 | invalid_input | invalid_input | 完整窗口，所有幀品質拒絕 |
| s4 | 已註冊p1 | 無推論結果 | 無推論結果 | 接受Start後camera open error |
| s5 | 未註冊 | review | timeout | 完整固定窗口 |
| s6 | 未註冊 | matched p2 | matched p2 | 完整固定窗口 |

必須算出：attempted=6、truth-known enrolled=4、unknown=2、paired-complete=5。A correct=1/4，B correct=0/4；兩臂wrong enrolled=1/4、unknown FA=1/2。s4在operation-error表且在E2E分母；s3不能因沒有usable frame而消失。A conditional paired-enrolled correct可以另列1/3，但不能拿1/3替換E2E的1/4。s1觸發T06、s2與s6觸發T01。B不能引用A的s1診斷matched而增加successful。

額外反例：同一s2追加一筆replay refusal，attempted仍為6；原始live結果不刪，新的replay run標refused且移出該run的paired集合。s2重播十次也不成十個真人session。相同participant的s1–s4仍是四attempt，另列participant=1、visit數按真實group，不將四次當四位獨立真人。

## 10. Exact-HEAD 原碼盤點與必修缺口

以下行號皆指本文件基線SHA，不是本機舊checkout。lead提供兩輪唯讀inventory後，commander獨立讀取關鍵原碼；只列原碼可支持的事實，不把尚未跑的runtime reproduction當已通過。

| 現況／locator | 對本計畫的影響 |
|---|---|
| `research/cli.py:235–250` 的 `cmd_live`已有fixed_seconds參數，但`:399–408`先跑到terminal，flag僅`_ = fixed_seconds`；`live/controller.py:201–207`得到terminal立即stop/release | **E3 must-fix**：目前真入口不能提供承諾的成功後完整5秒採集；不是加runbook就解決 |
| `research/cli.py:318–345`先open/probe/model setup，失敗直接return；recorder到`:360–364`才begin | **E1 must-fix**：研究嘗試帳本必須早於會失敗的camera/setup，否則分母只看到走進recorder的成功者 |
| `research/replay.py:167–195`用first captured作engine start、重新呼叫scorer；`live/frame_pipeline.py:304–312`用當下monotonic作processed；原start／processed未存 | **E4 must-fix**：不能保證原時間決策重演；需synthetic RED鎖定差異。不能直接把離線re-inference稱live reproduction |
| `research/replay.py:72–79`用frame cap **或**首末captured跨度標full | **E3/E4 must-fix**：完整窗口應由實際collection provenance判斷，非用滿25張推測；少於25張也可能採滿5秒 |
| `research/records.py:65–104`只存top1 ID/score/margin；`research/cli.py:143–160`margin=top1−runner_up | top2數值可在浮點容差內由差值推回；真正缺的是wrong-rank時truth score/rank、runner-up identity、原候選排序。不要為錯誤的「無top2不能分析」理由擴工 |
| `live/frame_pipeline.py:228–287`已有face count、geometry、quality量測；`:298–320`已有全identity scores；quality fail不embed | **E2**在既有量測點補診斷，不重算第二條pipeline、不繞quality gate；quality值多為proxy，landmark confidence在`:260`為常數1.0，不能稱實測confidence |
| `live/session.py:201–260`清支持／閾值／間隔／換身份；`:357–395`有品質最佳baseline，quality tie用較早sequence | **E2/E4**重用規則並暴露事件原因，不複製另一個決策器。identity score ties目前依mapping迭代順序，trace須保留順序／明確record tie；不要悄改tie-break |
| `research/report.py:113–116,181–184`逐outcome與逐refusal加attempted；`:186–190`以多profile版本推paired | **E5**改用明確attempt/run/arm鍵對帳；同attempt兩臂不能增加真人分母；同profile的A/B也應可配對。既有介面若提供重複輸入會重複計數，首次落地須以RED實際證明 |
| `research/report.py:118–123`used_for_tuning只重新分類；沒有凍結後才產生holdout的流程 | **E6**加prospective freeze protocol與read-time guard；不是把flag改名成sealed就有盲測 |
| `live/desktop.py:46–59,95–125,188–220`是headless `DesktopSession`；label_terminal僅改記憶體label；CLI`:420`傳None | **E1/E7**要有持久label revision的真正接線；**E7**最小真窗，不宣稱既有GUI已完成 |
| `scripts/live_preflight.py:62–81,103–106`無條件掃0..3並read；`:91–95`只有manifest數量與generation | preflight**會開相機**，不是camera-free。它不建立gallery digest、不驗models，exit0也不證明每台相機可讀，必須看JSON及實際chosen device |
| `scripts/live_teardown.py:25–41`傳device會開相機；`:65–66`檢查整個store/key-dir，並未以session篩選 | 多session研究store不能要求此腳本整庫為空；不得為了變綠刪除其他session。無device時camera為skipped，不是camera release PASS |

上述`research/`、`live/`皆以`src/facecore/`為前綴。相機腳本不在本輪執行。inventory第一稿的「refusal本身是漏洞」「同participant session應去重」「temporal fragmentation是首選根因」「preflight為camera-free」皆已撤回，不作計畫依據。

### 10.1 已核的既有介面

- `score_frame(frame: FramePacket, context: ScoringContext) -> FrameObservation`，`live/frame_pipeline.py:207`。
- `SessionEngine(profile, gallery_digest, model_generation)`；`start(session_id: str, now_ns: int)`；`observe(obs: FrameObservation) -> SessionResult | None`；`finish(now_ns: int, reason: str = "timeout") -> SessionResult`，`live/session.py:45–49,82,114,266`。
- `compute_baseline_best_quality(observations: list[FrameObservation], profile: ResearchProfile) -> tuple[FrameObservation | None, SessionStatus, str | None]`，`live/session.py:357–360`。
- `ResearchRecorder.append_frame(frame: FramePacket)`；`commit(result: SessionResult, frame_scores: tuple[FrameScore, ...] = ())`；`abort(session_id: str, reason: str)`；`revoke_image_consent(session_id: str)`，`research/recorder.py:240,284,327,335`。它的append只支援單一active session；本計畫不引入並行採集。
- `replay_session(bundle_id: str, *, store_root: Path, key_dir: Path, clock: Callable[[], datetime], profile: ResearchProfile, scorer: Callable[[FramePacket], FrameObservation], model_generation: str, gallery_digest: str) -> ReplayResult`，`research/replay.py:82–92`。
- `summarize(outcomes: list[LabeledOutcome], *, refusals: list[ReplayRefusal]) -> ResearchReport`，`research/report.py:100–104`。保留舊呼叫相容性，不把舊M3報表用新分母重新詮釋。

## 11. File／Interface／Dependency Map

### 11.1 工程檔案

**Modify（既有）：**

| 檔案 | 本次責任 |
|---|---|
| `src/facecore/research/records.py` | versioned attempt、label、trace、window／run結果契約；舊v1明確相容／拒絕能力，不假造新欄位 |
| `src/facecore/research/recorder.py`、`keys.py` | 新研究紀錄的AEAD、分離TTL、reconcile與session／participant刪除關聯；重用cipher不自製加密 |
| `src/facecore/research/cli.py` | 新研究入口／attempt前置、雙臂、分析與freeze接線；敏感內容不印stdout |
| `src/facecore/live/contracts.py`、`frame_pipeline.py`、`session.py` | truth-free diagnostics及decision events；原policy不變 |
| `src/facecore/live/controller.py`、`desktop.py` | fixed-window collector與inference-terminal分離；取消／錯誤／close仍有限清理 |
| `src/facecore/research/replay.py`、`report.py` | 真時間重演、同幀重推論的分流；attempt/arm/run指標 |
| `pyproject.toml`、lockfile、`.github/workflows/ci.yml` | 僅E7所需pyside6 optional依賴與headless/import smoke；確切lockfile路徑由lead在新base確認 |
| `docs/research/mac-live-runbook.md`、`docs/PROJECT-STATE.md`、`README.md` | 真命令、觀察值與限制；不宣稱計畫已實作 |

**Create（提案，不是已存在API）：**

| 檔案 | 責任 |
|---|---|
| `src/facecore/research/experiment.py` | ExperimentManifest、attempt/run identity、版本驗證 |
| `src/facecore/research/diagnostics.py` | 將truth-free observation／event序列封裝為研究trace；不含推論policy |
| `src/facecore/research/analysis.py` | evaluator-only rank/gap、trigger分類與case摘要 |
| `src/facecore/research/split.py` | candidate freeze、visit assignment、holdout release／污染紀錄 |
| `src/facecore/live/qt_window.py` | 薄pyside6視窗／方形採集與既有controller bindings |
| `tests/research/test_experiment.py`、`test_attempt_lifecycle.py` | schema、前置attempt、label與restart |
| `tests/research/test_diagnostics.py`、`test_diagnostic_privacy.py` | 完整trace、truth隔離、TTL／刪除 |
| `tests/live/test_fixed_window.py`、`test_qt_window.py` | 固定窗口、終局鎖定與真窗事件 |
| `tests/research/test_replay_clock.py`、`test_paired_analysis.py`、`test_split.py`、`test_phase2b_e2e.py` | 時鐘、雙臂分母、freeze與完整流程 |

lead盤點確認前四個新module在基線不存在。所有新檔在dispatch前再核collision；若新base已有同責任實作，重用而不是建立重複SSOT。

### 11.2 擬新增公共契約（E1先落，後續消費）

以下簽名是本計畫的**擬新增介面**，不得在E1/E2完成前當現有API呼叫。實作者若要改，須先由lead確認consumers/test同步，不自行換語義。

- `ExperimentManifest.from_dict(data: dict[str, object]) -> ExperimentManifest`；`digest() -> str`。欄位集合見§4；metadata敏感映射加密，不寫一般manifest JSON明文。
- `AttemptRecord`：experiment_id、attempt_id、participant_id、visit_id、condition_id、attempt_index、retry_of、consent_ref、requested/accepted/start/end時間、operational_status、error_code、bundle_ref；**不含truth**。accepted Start以durable加密attempt寫入成功為邊界；此前拒絕是start_denied，不冒稱已記錄的attempt。
- `EvaluationLabel`：attempt_id、revision、kind=`enrolled|unenrolled|uncertain`、identity_id（僅enrolled）、actor_ref、labeled_at；rev1之後更正保留audit。label只送evaluator，participant metadata不得被inference讀取。
- `ResearchRecorder.begin_attempt(manifest: ExperimentManifest, attempt: AttemptRecord, consent: ConsentRecord) -> None`；`finish_attempt(attempt_id: str, *, result: SessionResult | None, operational_status: str, error_code: str | None) -> None`；`write_label(label: EvaluationLabel) -> None`。原image staging begin/commit保持相容；attempt與image staging life-cycle分離，abort影像不抹掉已同意評估attempt。
- `FrameDiagnostics`（放`live/contracts.py`，truth-free）：原shape、normalized shape／capture transform、detector confidence／box／landmarks、quality各項實測或proxy值、stage durations；未產生的值為None＋reason，不填虛構0。
- `DecisionEvent`（同檔）：sequence、觀察是否接受、reset_reason、support_before/after、candidate_before/after、terminal transition、deadline餘量。SessionEngine新增可選keyword `event_sink: Callable[[DecisionEvent], None] | None = None`；預設None保留舊呼叫。事件來自原分支，不另寫一份狀態機推測。
- `FrameTraceEntry`：原FrameObservation的可序列化欄位、保留順序的identity-score pairs、FrameDiagnostics、DecisionEvent、staged_index或stage_missing_reason。**沒有label、姓名、pixels、crop或embedding。**
- `ResearchRecorder.append_trace(attempt_id: str, entry: FrameTraceEntry) -> None`；`read_trace(attempt_id: str) -> SessionTrace`。SessionTrace含schema、manifest digest、原start/deadline/end、collection_stop_reason、complete/incomplete、逐幀對映與B鎖定結果。
- `replay_observations(trace: SessionTrace, profile: ResearchProfile) -> SessionResult`：重用engine，保留原相對時序，沒有scorer或labels。
- `evaluate_arms(trace: SessionTrace, profile: ResearchProfile) -> tuple[ArmOutcome, ArmOutcome]`：A/B兩結果；ArmOutcome含attempt_id、run_id、arm_id、profile digest、selected/support sequences、terminal、collection extent、decision-time及refusal。不接受label。
- `analyze_batch(attempts: list[AttemptRecord], outcomes: list[ArmOutcome], labels: list[EvaluationLabel]) -> BatchAnalysis`：做§6/§8，attempt authoritative，outcome不得擴分母；無重新推論。
- `freeze_candidate(manifest: ExperimentManifest, *, code_sha: str, profile_digest: str, analysis_digest: str, planned_visit_ids: tuple[str, ...]) -> CandidateFreeze`；`authorize_holdout(freeze: CandidateFreeze, *, operator_decision_id: str) -> HoldoutRelease`。record custody由recorder持久化，純函數不自帶隱藏磁碟／網路副作用。

**serialization／migration：** 新study schema顯式版本化；舊v1 record可讀／刪／分析既有結果，但標`trace_unavailable`／`window_unproven`，不能補0假稱可精確重演。不能以schema升級重寫已測過的歷史報告。新增reader遇未知schema/AAD錯誤fail-closed。

**auth/data：** 無新網路認證／server；沿研究同意與本機key。participant-linked summaries/cases也屬敏感資料，刪某session後派生報告必失效／重算；撤回身份後其labels／trace／關聯聚合按批准的刪除規則處理。不得偷偷保留可反查的個資在「audit」名下。

## 12. 工程任務與PR邊界

這些是**G1之後才可派的工程包**，不是本輪執行清單。每包單獨可測／可merge，不疊在未merge的feature branch。一般preflight命令只用合成資料，不讀真人corpus。

所有包共同verification（由既有CI路徑核定；本次未執行）：

```bash
uv run --extra dev python -m pytest tests/ -q
uv run --extra dev ruff check src tests
uv run --extra dev mypy src
git diff --check
```

若new base的CI命令已改，lead在dispatch前以`.github/workflows/ci.yml`核實調整。每個behavior需immutable RED commit→minimal GREEN commit；最後exact HEAD review、CI、merge與post-merge gate走現行lead流程。涉及生命週期或資料邊界由lead評估dual/adversarial，不以「研究prototype」降級。

### E1／PR-1：讓嘗試與label完整落帳

**Depends：** D0＋G1。

**Files：** Create `research/experiment.py`、`tests/research/test_experiment.py`、`test_attempt_lifecycle.py`；Modify `research/records.py`、`recorder.py`、`keys.py`、`cli.py`。src前綴同§11。

**Consumes／produces：** 重用ConsentRecord、AEAD與現有record/image keys；產出ExperimentManifest／AttemptRecord／EvaluationLabel及§11 attempt/label方法。不更改辨識閾值。

**步驟：** 先寫schema／durable accepted-start與fault測試；將attempt建立移到任何camera open/model setup之前；image staging可以晚開始，但每個accepted attempt最終有operation result。將post-terminal label真正寫入加密sidecar，不再只存在DesktopSession記憶體。

**Acceptance：** camera-open／model-setup／start失敗、cancel、crash恢復仍各一attempt；相同ID重送不重複；record/image consent分離；label更正不改original inference；write失敗未接受Start不開camera；影像abort不刪合法的failure accounting；全同意撤回則依規則刪除且報告重新計算。不要用partially committed影像作計數權威。

**RED：** `uv run --extra dev pytest tests/research/test_experiment.py tests/research/test_attempt_lifecycle.py -q` → 注入open failure後應有attempt卻缺少，或label重啟遺失。先確認不是測試import錯誤掩蓋行為。

**GREEN：** 同命令全部通過；再跑`tests/research/test_cli.py tests/research/test_recorder.py tests/research/test_recorder_recovery.py`及共同verification。

**PR-1驗收出口：** 無相機即可證明attempt守恆與label持久化；影像／trace尚未完成不能宣稱TOOLING_READY。

### E2／PR-2：可定位失敗的加密trace

**Depends：** E1。

**Files：** Create `research/diagnostics.py`、`tests/research/test_diagnostics.py`、`test_diagnostic_privacy.py`；Modify `live/contracts.py`、`frame_pipeline.py`、`session.py`、`controller.py`、`research/records.py`、`recorder.py`、`keys.py`。

**Consumes／produces：** 原score_frame／FrameObservation／engine分支；產出FrameDiagnostics、DecisionEvent、FrameTraceEntry、SessionTrace與append/read_trace。既有`score_frame(frame, context)`保留；可增加keyword-only diagnostic sink，default None。

**步驟：** 在既有detect/quality/score點取原值；identity scores以原迭代順序存pairs，避免serialize sort_keys後改tie結果。engine每個clear/skip/terminal分支發event；record observation、controller deadline、stage index以及staging錯誤，不吞例外假裝bundle完整。

**Acceptance：** wrong-rank可在evaluator重建truth score/rank；quality fail留原因／可用metrics且embedder呼叫數=0；None runner-up、interval skip、score reset、identity change、continuity、late processing各有唯一事件。trace有界≤25 sampled observations，無pixels／embedding。diagnostic關閉與開啟在injected clocks的終局一致；真機另量IO/trace overhead，不保證現場時序零影響。

新trace包含score/ID與geometry，視敏感研究資料；評估trace沿record TTL，原像素沿image TTL；任何影像可逆衍生物改走image TTL。新增keys/blobs/labels/cases的刪除、reconcile、tamper、cross-session AAD替換、7/30天分離都需測。儲存失敗產生operation error／window incomplete；若inference早已鎖定仍保留其原terminal，不能覆寫歷史或冒稱成功保存。

**RED：** `uv run --extra dev pytest tests/research/test_diagnostics.py tests/research/test_diagnostic_privacy.py -q` → 無quality/reset trace或刪後仍可讀新blob。

**GREEN：** 同命令；加`tests/live/test_frame_pipeline.py tests/live/test_session.py tests/research/test_recorder_recovery.py`及共同verification。

**PR-2驗收出口：** 每個§8 signature有可觀測欄位或明確missing reason；尚未補固定窗口不得採正式對照資料。

### E3／PR-3：真正固定窗口，不破壞取消與worker清理

**Depends：** E2。

**Files：** Create `tests/live/test_fixed_window.py`；Modify `live/controller.py`、`desktop.py`、`research/cli.py`、`records.py`、`recorder.py`；既有回歸`tests/live/test_pump_release_race.py`、`test_controller_integration.py`、`tests/research/test_cli_lifecycle.py`。

**Consumes／produces：** 既有`--fixed-seconds`、engine及trace；產出真實collection_start/deadline/end、stop_reason、complete flag與完整共同observation流。

**最小行為：** 將「B inference terminal」與「collector terminal」分開。普通mode仍早停；fixed mode中B第一次terminal鎖定後，只有已同意影像保存且安全的session才能持續至原deadline；後續幀可供A與診斷，不能再送B改結果。25幀／5fps上限保持，滿25幀也不捏造5秒時間證據。

**Acceptance：** 0.4秒B matched後仍收集到原5秒邊界；A能看到後段更高品質幀；取消／close／撤回／多人／continuity不明／error即刻停止，collection incomplete。提前鎖定後仍要有安全監測，不能因B不再observe而漏掉多臉／撤回。即使B早已matched，後續collector錯誤也要另列，不把不完整流當full。stop→join→release維持，不能重引入close-during-read競態。

**RED：** `uv run --extra dev pytest tests/live/test_fixed_window.py -q` → 通過真CLI/controller caller的fake clock＋fake source案例，現no-op只收early流，斷言失敗。不要只測新collector helper而漏掉CLI wiring。

**GREEN：** 同命令＋`uv run --extra dev pytest tests/live/test_pump_release_race.py tests/live/test_controller_integration.py tests/research/test_cli_lifecycle.py -q`＋共同verification。race類依fleet獨立RED/GREEN及三次穩定重演。

**PR-3驗收出口：** complete/incomplete由collector證據決定；native真機Stop仍須E8當場驗，synthetic綠不代替。

### E4／PR-4：原時間重演與雙臂配對

**Depends：** E2＋E3。

**Files：** Create `tests/research/test_replay_clock.py`；Modify `research/replay.py`、`cli.py`、`records.py`；既有回歸`tests/research/test_replay.py`、`test_wired_frames.py`、`tests/live/test_session.py`。

**Consumes／produces：** SessionTrace與已核的compute_baseline_best_quality、SessionEngine；產出replay_observations／evaluate_arms／ArmOutcome。既有replay_session保留為明示的re-inference用途，不能繼續宣稱精確live重演。

**Acceptance：**

- 原start=0、第一幀晚到、processed跨deadline的合成案例重演原B結果；不能以first-frame timestamp偷換session start，不能用replay wall clock。
- sequence不連號是合法drop；staged index不是sequence。中間blob缺失、ledger/blob對不上、time回退、非法重複seq拒絕並列原因，不將後幀左移補洞。
- fixed collector真採滿5秒但只有10幀仍可full；25幀在短流或unknown coverage不能full。cancel/error依其結束原因incomplete，不以frame count洗白。
- A對整個共同合法窗口呼叫原baseline helper；B對原observation重演首次terminal，兩者相同profile並以arm_id區分。single-id/None margin仍不可matched。
- serialized identity ties可重演；label變更不改任一arm。frames_read、frames_scored、B_consumed、staged分開計數，避免現行read全blob但early-break仍稱全部replayed的混淆。
- 舊無trace bundle標unproven，只能legacy診斷；model變更研究需要G4新generation gallery及獨立re-inference方案，不能為重播強行關generation gate。

**RED：** `uv run --extra dev pytest tests/research/test_replay_clock.py -q` → 原processed late而重演early-success、start漂移、假full或mapping錯誤案例之一失敗。

**GREEN：** 同命令＋`uv run --extra dev pytest tests/research/test_replay.py tests/research/test_wired_frames.py tests/live/test_session.py -q`＋共同verification。

**PR-4驗收出口：** 原decision exact重演与同幀重推論分清；detector浮點差異不能靜默改成「等價terminal」，有差異觸發T02。

### E5／PR-5：分母、診斷分類與每批分析入口

**Depends：** E1＋E2＋E4。

**Files：** Create `research/analysis.py`、`tests/research/test_paired_analysis.py`；Modify `research/report.py`、`cli.py`；回歸`tests/research/test_report.py`、`test_main_entry.py`。

**Consumes／produces：** attempt authoritative set、ArmOutcome、EvaluationLabel；產出BatchAnalysis及§8 trigger／§9 case template。擬新增CLI：`python -m facecore.research analyze --store S --key-dir K --experiment X --mode development`；**E5落地並核help前不可當現有command執行。** 詳細case寫AEAD store，stdout僅aggregate與run code。

**Acceptance：** §9.1六列全數精確對帳；兩arm或同attempt多run不擴attempted；report合法處理uncertain／無label／unknown／None score／zero denominator。每arm分開refused，不把缺資料當unknown。top1錯／門檻擋／支持不足由trace分類，缺證據輸出UNRESOLVED_EVIDENCE。每個known failure有trigger或明確「單次觀察，等待同signature再現」；hard trigger不得被overall平均掩蓋。

保留舊summarize caller相容性；2B必經新attempt-aware入口，不能由caller先過濾error再交report。數據分析不開相機、不跑embedder、不更新gallery。沒有eligible row時不崩潰也不印0%冒充準確率。

**RED：** `uv run --extra dev pytest tests/research/test_paired_analysis.py -q` → 六列算例、outcome+refusal同ID、同profile雙arm、label-change isolation斷言失敗。

**GREEN：** 同命令＋`uv run --extra dev pytest tests/research/test_report.py tests/research/test_main_entry.py -q`＋共同verification。從CLI到report的integration至少一條，不僅測聚合helper。

**PR-5驗收出口：** operator能讀到「哪一層失敗／下一個控制」而不只一個成功率；仍不會自動修改參數。

### E6／PR-6：prospective holdout與candidate freeze

**Depends：** E1＋E5。

**Files：** Create `research/split.py`、`tests/research/test_split.py`；Modify `research/experiment.py`、`recorder.py`、`cli.py`、`analysis.py`。

**Consumes／produces：** §11 CandidateFreeze／HoldoutRelease、planned visit IDs、hashes；輸出可稽核且不可靜默改寫的freeze／release／contamination紀錄。

**最小路線：** candidate凍結後才採未來visit；不要為此建立大型RBAC／服務。正式分析入口在holdout release前拒絕content分析，舊replay／read入口也不得讓已註冊sealed bundle繞過同一research guard。這是防誤用流程，不宣稱抵抗持本機key的管理者；需要預存sealing抵抗分析者存取時須另有custodian與OS/key隔離。

**Acceptance：** 同visit跨split、已用development標holdout、freeze後換candidate/hash、未release分析、重覆同次holdout當新驗證均拒絕；原資料不刪、污染有紀錄。合法開封後可重跑核算相同凍結結果，但不能選新策略並稱第二次未見測試。這裡的「只評一次」是一次確認性分析，不禁止同條件reproducibility audit。

**RED：** `uv run --extra dev pytest tests/research/test_split.py -q` → 上述非法轉移被接受或可經legacy replay讀取。

**GREEN：** 同命令＋E5 integration與共同verification；truth仍只在evaluator。

**PR-6驗收出口：** 新visit的採集先後順序和凍結hash可查；無實際參與者也可synthetic驗流程，不冒稱已有holdout。

### E7／PR-7：最小pyside6引導窗與方形採集

**Depends：** E3＋E5＋E6。

**選擇：** 本計畫建議**正式2B採集前完成最小真窗**；CLI僅工程／synthetic演練。這避免「人工方形框紀律」代替已批准spec。若operator想先做CLI真人診斷，必須另明示縮限scope／接受採集方式差異，不能在實作時自行豁免。

**Files：** Create `live/qt_window.py`、`tests/live/test_qt_window.py`；Modify `live/desktop.py`、`research/cli.py`、`pyproject.toml`、lockfile、`.github/workflows/ci.yml`、runbook；回歸`tests/live/test_desktop.py`、`tests/research/test_camera_wiring.py`。

**Consumes／produces：** 原DesktopSession與E3 collector API，E1 label持久化。擬新增`live --ui qt`路由；既有fake/CLI保留。不要把Qt import變為核心library必需，明確research-ui optional extra。

**非真人spike先行（上限半工作日）：** 核pyside6對本專案Python/macOS環境的wheel、授權與主執行緒事件循環；以synthetic preview驗timer／close／worker join，記確切版本與命令。若新依賴不可行就BLOCKED回operator，不換Cocoa／web。spike與產品實作分開派工，結果落同一dependency manifest供PR-7沿用，不杜撰未測可用版本。

**最小UX：** 裝置確認、研究非認證說明、雙同意與TTL、方形框、Start/Cancel、collection倒數／保存狀態、terminal後label及刪除。fixed mode早期B terminal後顯示「辨識已鎖定、仍在採集至截止」，Cancel能停止剩餘採集；不能因Desktop已terminal而拒絕停止錄影。

**採集幾何契約：** 原sensor frame經orientation規範後，以`S=min(W,H)`、左上`((W-S)//2,(H-S)//2)`中心方形裁切，**不非等比縮放**；overlay與實際crop使用同一mapping，mirror只preview。這是本輪擬定capture contract，D0需寫入spec附錄。保存的已同意原sample frame與crop mapping可重建同一inference輸入；若只保存square input，必須在manifest明示不能回看框外，不冒稱保留sensor全幀。第一輪固定其中一種：**沿既有有限全幀保存，另記crop mapping；不額外保存第二份crop。** 完整背景仍受單人受控場地與consent約束，不能宣稱crop外的人必被偵測。

**Acceptance：** synthetic棋盤格portrait/landscape／mirror overlay與crop對位，identity不參與crop；640 letterbox／112 align不因capture square而取消。UI操作到真controller/recorder/label接線；quality-rejected不embed、uncertain不顯猜名；close與cancel含B早期matched後皆停止／join。資料預覽不落截圖。

**RED：** `uv run --extra dev pytest tests/live/test_qt_window.py -q` → fake Qt events驅動真bindings時shape/cancel/label持久化缺失；不能以整檔skip算過。

**GREEN：** Qt依賴spike凍結後，以該extra環境跑同命令與`tests/live/test_desktop.py tests/research/test_camera_wiring.py`；CI至少import＋offscreen synthetic實測。operator在場另驗真窗，不把offscreen當macOS權限證據。

**PR-7驗收出口：** 能以一致的方形採集與明確同意完成研究；不增加美化dashboard、模型選擇器或live learning。

### E8／PR-8：串起真入口的研究啟用驗收與runbook

**Depends：** E1–E7。

**Files：** Create `tests/research/test_phase2b_e2e.py`；Modify runbook、PROJECT-STATE、README及必要的既有integration tests。不加另一條evaluation pipeline。

**Acceptance：** 真CLI／Qt Start→fake camera→真score adapter(test doubles)→真engine→AEAD store→label→雙臂→analysis→freeze→未來synthetic visit→holdout→刪除／重啟不可讀全鏈。至少包含§9.1六列、late/reset/missing blob、failed attempt、holdout contamination、record/image expiry、收集早期matched後取消。每個trigger有一個合成反例驗證分類不是裝飾。

**RED：** `uv run --extra dev pytest tests/research/test_phase2b_e2e.py -q` → 任一wiring故障使endpoint輸出對不上預期；不是直接注入已整理好的outcomes省略前半條鏈。

**GREEN：** 同命令＋完整suite／ruff／mypy／diff check；UI extra/import測試沿E7凍結方式。runbook寫明各exit code、JSON欄位與manual checkpoint，不以exit0代替read成功。

runbook同時補上目前零引用的`scripts/live_preflight.py`與`scripts/live_teardown.py`（盤點缺口#6），並照§14標明preflight會開相機、teardown掃整個store/key-dir、`--device`才碰相機。這是把既有腳本正確標示，不是改寫腳本行為；要改行為另立最小task。

operator在場非真人相機smoke：確認requested與resolved device／實際shape、原deadline、camera stop/reopen、GUI close、隔離synthetic store刪除。真相機preflight只有在operator確認場地且允許開相機時跑；不能把「沒保存」理解為可隨意啟動camera。

**PR-8驗收出口：** `TOOLING_READY`且所有未驗硬體項清楚列出。disconnect與OS權限層若仍未驗，維持retained，不重開2A也不冒稱已驗；若本輪實際碰到其故障則阻擋相關採集直到最小處置。

## 13. Dependency、派工與驗收順序

D0 → G1 → E1 → E2 → E3 → E4 → E5 → E6 → E7 → E8 → G3 → R1/R2 → 必要時R3 → G4 → 單一candidate工程 → development核驗 → G5 → H1 → G6。

E7的非真人dependency spike可在E2之後獨立準備，但spike需自己的go/task且不擋E3–E6的camera-free工程。所有PR獨立merge到當前main，不為加速疊未merge分支；共享檔由lead串行整合。若某已有功能在新base已補齊，lead用相同acceptance證據關掉該工作包，不為照表重造。

**每次派工brief必帶：** parent研究task、唯一assignee、source spec與本plan版本、live base/head、該E包範圍與排除、depends_on、RED/GREEN、privacy／real-camera界線、expected report。異常回該次dispatcher，不覆寫operator對lead的直接工作令。

**每包report：** 已變檔案、exact SHAs、實際commands/exit/failure、schema/caller相容性、unknowns、下一包能消費的產物。非PR研究batch報告不假造PR／CI里程碑。對runtime／精確replay的主張需runtime evidence；本計畫的source inspection只是制定測試依據。

## 14. 命令卡：哪些會碰相機／敏感資料

所有相對路徑命令均在lead／impl自己的daemon-bound worktree內執行；此輪只讀source，**以下沒有被執行過**。

| 命令 | 類別／限制 |
|---|---|
| `uv run --extra dev python -m facecore.research --help` | 現有雙入口help；不當作2B功能已存在的證據 |
| `uv run --extra dev pytest tests/live tests/research -q` | 既有synthetic測試；仍先核new base tests是否新增硬體副作用 |
| `uv run --extra dev python scripts/live_preflight.py --help` | 只讀parser；不跑checks |
| `uv run --extra dev python scripts/live_preflight.py --corpus "$CORPUS" --models "$MODELS"` | **會開相機**、讀manifest；G3/在場批准後才跑。models目前parser接收但不實際驗證，gallery digest須由研究context本身取得 |
| `uv run --extra dev python scripts/live_teardown.py --store "$ISOLATED_STORE" --key-dir "$ISOLATED_KEYS" --session "$SESSION"` | 不傳device所以不開相機；檢查整個指定store/key-dir，僅適合該次隔離驗收，不代表相機已釋放 |
| 同上追加`--device "$RESOLVED_INDEX"` | **會開相機**驗reopen；不能傳`local`給只接受int conversion的script；read欄位也要核，不只看exit |
| `uv run --extra dev python -m facecore.research.cli delete --store "$STORE" --key-dir "$KEYS" --session "$SESSION"` | 真刪除、需相應授權；只刪指定session，研究批次刪除要按到期／撤回清單，不為teardown變綠刪全庫 |

`$CORPUS/$MODELS/$STORE/$KEYS/$SESSION/$RESOLVED_INDEX`不是預設值；由G3已批准manifest解析。新`analyze`／`--ui qt`及freeze入口在對應E包落地後，以`--help`和integration test核新語法；未落地前禁止照貼。

## 15. 研究結果的交付與完成判定

每輪交付三件，不交巨量未解讀log：

1. **Batch card：** freeze refs、participant/visit/attempt數、truth缺失、每臂counts/denominators、完整／不完整窗口、正確／錯認／unknown FA、time與quality/drop摘要、coverage和排除列表。
2. **Case packets：** 所有T01–T03及需要RCA的signature，依§9寫到能反駁／定位；原資料locator與expiry由加密store保存。
3. **Decision memo：** 已知、被否定、未解；下一步唯一owner，需哪個go、成本／新資料需求、不能宣稱什麼。

最後只能擇一：`BASELINE_CHARACTERIZED`、`CAUSE_SUPPORTED`、`IMPROVEMENT_SUPPORTED_LIMITED`、`NO_IMPROVEMENT`、`INCONCLUSIVE`、`BLOCKED`，並列其適用範圍。只有operator可另宣告「這個Mac研究用途已足夠可用」；不能從小樣本沒有FA升格部署。

### 15.1 Requirement coverage

| Operator需求 | 對應交付／驗收 |
|---|---|
| 解決Mac辨識度問題，不只工具完成 | §5 R1→R4→H1、§6改善判準、§15明確不成功出口 |
| 取得數據即有深入分析觸發 | §7最早阻斷層、§8 T01–T12、E2/E5/E8可測 |
| 中等能力模型可接手 | §0先讀、§3 gates、§9 case格式／六列算例、§11介面、§12每包RED/GREEN、§14命令風險 |
| 大量盤點交lead | 子task `t-20260916004222455161-53424-5`；原碼關鍵結論由commander另核，不把摘要當事實 |
| 不越權啟動實作 | G0/G1/G3/G4/G5分開；本輪只交文件，未派上述E/R/H任務 |

### 15.2 仍需operator在執行前決定，不由模型猜

- G0/G1：是否接受配套spec與本計畫、是否授權D0文件landing及首波工具工程。可分次批准。
- G3：參與者／各自同意、非同步store/key位置、TTL、起始profile來源與值、實際相機、上述最多24 development＋12 prospective holdout的預算是否適合。數量不足可以診斷，但不補造open-set驗證。
- G4：哪一個具證據的改善提案可實作。此刻沒有新session，不能先選threshold／模型／template方法。
- G5：candidate凍結後，何時安排新visit或採獨立custodian的sealed方式。預設建議prospective，非強迫收更多資料。

這些是**有明確owner、gate及停止動作的外部批准項**，不是讓實作者自由填的技術空格。

## 16. 接手清單與最高風險

開始一個包前：讀live board／spec、核base drift、確認該包go、核bound worktree、讀source與tests、先RED；遇到缺失前提先report，不「順手」擴工。結束包後：貼exact evidence／已知限制、依lead流程收斂，不把本plan回填成永久歷史文件。

最高風險依序：**不公平或失真的資料管線**（fixed-window／clock／分母）、**有資料但無法定位**（truth-rank／reset／quality未留）、**小樣本與holdout污染**、**錯認被平均成功率掩蓋**、**trace擴大隱私面／TTL刪漏**、**UI/fixed mode重引入相機生命週期競態**。每一項已分別由E3/E4/E5、E2、E6、T01、E1/E2、E3/E7/E8設gate。

**本文件完成≠工程完成；工程完成≠辨識改善；development改善≠新資料成立；新資料有限成立≠部署核准。**
