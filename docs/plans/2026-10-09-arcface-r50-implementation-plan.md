# ArcFace R50 接入實作計畫

**Source of truth:** `d-20261009075738925782-1`、`d-20261009080904985166-2`；implementation task `t-20261009075814821312-51495-5` v2 dispatch。
**Goal:** G3 真實入口以 hash gate 載入 ArcFace R50 與專用 profile；錯配與模型失敗皆拒絕，SFace 歷史評估不變。
**Architecture:** 保留 `_build_true_context` 的 SFace default，由 `cmd_live` 提供 ArcFace default factory。`Embedder` 依 manifest 前處理並讀取 session IO；入口依實際模型身分驗證 profile 配對。gallery 每次 App 啟動重建，不遷移持久化 template。
**Tech stack:** Python、NumPy、ONNX Runtime、pytest。

## 全域限制

- 權重、照片、embedding 與實測分數不進 Git；外部資料只讀。
- 不改 align3、UI、eval、production governance／lifecycle 或 Phase-1A CLI。
- 保留 `profiles/g3-v1.json`；新 profile 除身分 metadata 與授權 match threshold 外沿用其值。
- model version `arcface-w600k-r50-fp32`；generation `arcface-112-rgb-minus127.5-div127.5+align3`。

## Evidence 與待取得證據

- Established：`research/cli.py:_build_true_context` 建立 in-memory gallery；`frame_pipeline.py:267–271` digest 輸入為 identity 與 embedding bytes。spike `t-20261009080005883613-51495-6` 已完成。
- Established：D5／D6 反向共用 `_build_true_context`，不能改其 SFace default；operator reference 使用 repo align3。
- 待取得：實際 manifest＋Embedder 對 reference 的逐張等價、mobile checker 完整輸出。owner 為 implementer；超出容差且原因不在 scope 時停止回報 Lead。

## 文件影響前置檢查

`docs-check: arch=N adr=Y area=Y`。已載入 `project-docs-maintain`，此檢查為 implementation 前置 dependency。
搜尋 `g3-v1|SFace|sface|model` 已確認 `scripts/g3-local-test-app.command:100`、`docs/g3-local-test-sop.md:64` 與 spec:31 的 G3 操作契約需同步。Lead 已核准 SOP 改新 profile，spec 保留原文並新增日期註記。歷史 D0／D5／D6 基準不改。

## 檔案與依賴

- Modify `src/facecore/contracts/manifest.py`：ArcFace factory、前處理／redistribution metadata，既有 factory 預設值保持相容。
- Modify `src/facecore/pipeline/embed.py`：manifest 前處理與 session IO；SHA gate 先於 inference。
- Modify `src/facecore/research/cli.py`：G3 default、實際模型/profile 配對、gallery generation。
- Modify `src/facecore/live/contracts.py`：profile backend metadata 的相容載入。
- Create `profiles/g3-v1-arcface-r50.json`；Modify launcher 明確指定新 profile。
- Create `tests/pipeline/test_arcface_embed.py`、`tests/research/test_arcface_live.py`：synthetic contract 與真入口測試；真權重依既有 device-gated 慣例。
- Create ADR 0011、research 授權紀錄；Modify PROJECT-STATE、G3 SOP／spec 日期註記，互相以 pointer 連結。

## Task 1：RED 契約

**Consumes:** `ModelManifest`、`Embedder(manifest, artifact_path)`、`cmd_live`、`_build_true_context`、`ResearchProfile`。
**Produces:** 先於實作的 immutable RED commit。
**Cases:** ArcFace normalized RGB NCHW＋動態 IO；manifest 欄位填實；真入口 default 缺失／錯誤 hash 使用 model-failure 理由且無 fallback；injected factory 雙向錯配拒絕；helper default 保留 SFace；ArcFace generation 不用 gen-1。
**RED/GREEN:** `uv run --extra dev pytest tests/pipeline/test_arcface_embed.py tests/research/test_arcface_live.py -q`。RED 應因缺少 factory／入口 guard／ArcFace default 而失敗，不接受環境或 setup error 作證據。GREEN 同命令全部通過。

## Task 2：最小實作

依 Task 1 的測試建立 manifest 與前處理；保留 SFace RGB raw NCHW bytes 的數值行為。只在 `cmd_live` 生產 default 注入 R50 factory；配對檢查依實際 model version 與 profile metadata，不依 profile 檔名；注入 factory 經相同 guard。模型 setup 例外使用既有 `ModelSetupError`／啟動失敗路徑。

## Task 3：真權重與文件驗證

- repo 外 isolated checker 環境：`onnxruntime==1.30.0`＋確切 onnx 版本；實際命令與輸出寫入 manifest／research。用後刪環境並回報刪除證據；不修改 product dependencies、lock 或 canonical venv。
- 實際 manifest＋Embedder 在 operator reference 的同一照片母體逐張比對，embedding absolute difference 與 cosine error 均不超過 dispatch 容差；數字僅回報。
- ADR 0011 記非商用自用、權重不散佈、兩則 decision locator；已知 alignment 限制只放 spike locator，不放量測數字。
- PROJECT-STATE 以 ADR／research／plan 為 pointer；不改歷史 frozen baseline。

## Task 4：交付

產品驗證命令由 `pyproject.toml` 與 CI 的 live baseline 確認後執行；包括既有 SFace tests、全 pytest 與既有 lint/type checks。驗證 diff 不含 exclusions；建立 ready PR，回報 RED/GREEN full SHAs、PR locator、actual commands/output、deviations、known issues；CI 成功須由 provider 驗證。

## PR 邊界

本 task 按 Lead v2 dispatch 為單一 G3 backend 替換 PR：manifest、前處理、入口配對、profile／launcher 與文件一起形成可啟動且不錯配的狀態。RED tests 先獨立 commit，GREEN implementation 後續 commit。交付後停止 mutation，等待 Lead review／rework／merge。
