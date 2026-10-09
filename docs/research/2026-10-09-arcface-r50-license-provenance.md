# ArcFace R50 授權與 provenance 審查

日期：2026-10-09。範圍與用途由 [ADR 0011](../decisions/0011-arcface-r50-g3-default.md)／`d-20261009075738925782-1` 定義；本文件不宣稱商用或再散佈授權。

## Code 與 weight 授權分開記錄

- Code：上游 [Python package README License](https://github.com/deepinsight/insightface/blob/master/python-package/README.md#license) 原文：「The library code is released under the **MIT License**, for academic and commercial use.」本專案使用自己的 ORT adapter，不新增 InsightFace runtime dependency。
- Weight：[model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) 原文：「ALL models are available for non-commercial research purposes only」。
- 本專案依 operator decision 限自用、非商用、不散佈；redistribution：無，權重不進 Git、不散佈。用途改變須先取得適用授權。
- Provenance：model zoo 的 `buffalo_l` recognition model 標示為 `ResNet50@WebFace600K`。此為上游訓練資料來源聲明，不是訓練資料個別同意或權利清查。

## Artifact

- 上游 pack locator：[buffalo_l](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)。未在本次工作下載或散佈權重；對既有本機檔作唯讀驗證。
- 本機路徑：`~/facecore-models/w600k_r50.onnx`。
- 本機查核日期：2026-10-09（不推定原始下載日期）。
- `shasum -a 256 ~/facecore-models/w600k_r50.onnx`：`4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`。
- Manifest：`ModelManifest.arcface_w600k_r50_fp32()`；載入前以 exact SHA gate 驗證，沒有 inference 時的網路下載。

## Mobile usability 實際執行 provenance

在 repo 外的拋棄式環境執行，版本為 Python 3.11.14、onnxruntime 1.30.0、onnx 1.23.2。未修改產品依賴、lock 或 canonical venv。直接 `python -m ...usability_checker` 無 logging handler，未把其空輸出當成分析成功；以下命令明確啟用 logging。

```sh
/tmp/arcface-checker-20261009/bin/python -I -c 'import logging,sys; logging.basicConfig(level=logging.INFO); from onnxruntime.tools.mobile_helpers.usability_checker import run_analyze_model; sys.argv=["checker",sys.argv[1]]; run_analyze_model()' "$HOME/facecore-models/w600k_r50.onnx" > /tmp/arcface-mobile-checker-20261009.log 2>&1
```

原始輸出：`/tmp/arcface-mobile-checker-20261009.log`；SHA-256 `a2b1ab14e73485eec578b2967d5c66d4e987e303e1a52e4fe0a6ba220ca8cfc8`。完整結果摘要保存在 manifest 的 `mobile_usability`（含版本與可重跑命令）；原始輸出不放 Git。

Checker 判定原始 dynamic-shape 模型的 NNAPI／CoreML NeuralNetwork／MLProgram 均為 NO；假設固定 shape 時，NNAPI／NeuralNetwork 為 YES，MLProgram 仍為 NO，unsupported operators 為 BatchNormalization／Flatten。這是 checker 假設分析，本次沒有修改或輸出 fixed-shape 權重，也不是行動端實機證據。

拋棄式環境於同日刪除；刪除命令先確認 `pyvenv.cfg`，刪除後確認路徑不存在。模型 SHA 仍與上列一致。

## 等價驗收 provenance

- 對照來源：`~/Downloads/face_sample/測試ArcFace離線照片評估.command` 與 `run_g3_arcface.py`，唯讀；只採 R50 repo-align 前處理，不採 MBF fallback。
- 實際 manifest＋Embedder 量測輸出：`/tmp/arcface-manifest-equivalence-20261009.out`，2026-10-09；SHA-256 `51f5c3b37ec819f4a1319a5e27eca412c8bd49831abaf69cc35f82994e0ceaa4`。
- 拋棄式量測 script：`/tmp/arcface-premise-20261009.py`；SHA-256 `8e7db66e4110b934ca86548565f217641cdc82e357f239232ab1f5b083bebaca`。
- 分數與逐張結果只交付 task report，不提交 Git；`/tmp` 不是永久 evidence store。限制見 ADR 0011。
