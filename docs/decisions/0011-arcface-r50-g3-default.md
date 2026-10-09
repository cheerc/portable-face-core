# ADR 0011：G3 預設採用 ArcFace R50

- 狀態：accepted，限自用、非商用、不散佈。
- 日期：2026-10-09
- 上位決策：`d-20261009075738925782-1`；實作 fork：`d-20261009080904985166-2`。

## 決策

G3 的 `cmd_live` 真實 collection default 使用 InsightFace `w600k_r50.onnx`，model version 為 `arcface-w600k-r50-fp32`。launcher 明確選取 `profiles/g3-v1-arcface-r50.json`。政策設定以該 profile 為 source of truth，不在文件重述實測分數。

YuNet 與 align3 不變。ArcFace 前處理為 RGB、減去並除以 manifest 的 normalization 常數、NCHW float32、embedding L2 normalization；IO 名稱由 session 讀取。載入前驗證 manifest SHA，缺失或不符採既有 model-failure 啟動理由，禁止 fallback。

模型與 profile 的配對檢查在 gallery 建立前使用 embedder 的實際 model version；injected factory 也走同一道檢查。ArcFace profile 明示 `embedding_model_version`，其 generation 為 `arcface-112-rgb-minus127.5-div127.5+align3`。舊 profile 未提供該 metadata 時保持原序列化／digest。

G3 每次 App 啟動從照片重建 in-memory gallery；不遷移既有 production template。`_build_true_context` 的 SFace default、`g3-v1.json` 與 production governance／lifecycle 保持不變，供歷史評估使用。舊 session 不重新詮釋為新模型資料。

## 授權與 provenance

Model zoo 原文：「ALL models are available for non-commercial research purposes only」。operator 依上位 decision 定性本專案用途為自用、非商用、不散佈；不將此裁定解讀為商用或對外散佈授權。權重不進 Git、不散佈，亦無 runtime 自動下載。若用途改變，先另行取得適用授權。

詳細 code／weight license、artifact locator、checksum、WebFace600K provenance 與 mobile checker 紀錄，見 [授權審查](../research/2026-10-09-arcface-r50-license-provenance.md)。執行與驗收邊界見 [實作計畫](../plans/2026-10-09-arcface-r50-implementation-plan.md)。

## 已知限制

- repo 外 `~/Downloads/face_sample/run_g3_arcface.py` 使用未知模型標籤 `arcface_r50_v1`，會被入口拒絕；merge 後改用 repo 的 `scripts/g3-local-test-app.command`。不修改該外部 runner，也不為測試另開 guard 例外。
- repo align3 與上游 InsightFace `norm_crop` 不作數值等價宣稱；上游 exact runtime 未驗。locator：spike `t-20261009080005883613-51495-6`／report `m-20261009080828718465-336`。本次等價驗收以 operator 的 repo-align reference 為準。
- Mobile checker 不等於 Android／iOS 實機驗證；不量化、不修 graph、不宣稱行動端效能。
- 本機 reference 重現不證明代表性人群準確率、低 FAR、liveness 或安全認證。
