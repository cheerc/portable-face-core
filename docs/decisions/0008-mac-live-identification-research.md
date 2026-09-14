# ADR 0008：先做 Mac 動態辨識研究，再決定 mobile 與部署

- 日期：2026-09-14
- 狀態：operator 已同意設計方向；本書面決策待 review／merge。僅文件授權，實作另需 plan＋go。

## Context

靜態 23 人 gallery、13 張本人探針與 30 張 non-target 是現有研究資料，不能替代動態掃臉的 session 級評估。先手工建大圖庫才准研究，會增加使用者負擔；直接建 Android/iOS 又過早投入平台工程。

## Decision

新增 **Phase 2A：Mac 動態辨識研究原型**，先於 mobile prototype。單照註冊、單人相機引導、有界多幀 session、獨立評估標記與同意後回放保存，用於形成實驗資料與比較策略。正式契約集中於 [研究設計](../specs/2026-09-14-mac-live-identification-research-design.md)，本 ADR 不重複參數。

- 相機/UI 是 adapter；session engine 是新研究層，不能將離線 replay harness 冒充即時追蹤器。
- 多幀相似度判斷可獨立研究，不需先完成 authentication layer；它仍不提供 liveness、防重播或 `authenticated`。
- 原型第一版固定 gallery、不更新正式模板。研究保存同意與 learning confirmation 是不同權限。
- 明列獨立研究 recorder 的有限全幀保存例外：事前同意、加密、TTL、刪除、無雲端；不變更 Face Core 的 no-full-background-retention 或 zero-disk-before-learning-confirmation 契約。
- 原 spec 的 Phase-2 representative-gallery gate 適用正式多身份部署／目標容量宣稱；cross-runtime gate 適用 mobile 評估。允許先做受控 Mac 研究以建立 evidence，並非降低部署／mobile 門檻。
- 原 spec §15 的 video recognition 未研究限制，僅對本設計範圍開放研究；不限時監控、video-driven 正式模板更新與 mobile pre-capture 錄影仍不包含。

## Alternatives

繼續純照片：重現容易但使用情境不符。直接 mobile：使用情境接近但迭代負擔大。採 Mac 研究原型，以小批真實 session 自然累積語料，減少手動圖庫整理。

## Consequences

Phase 1A/1B 已交付工程不用重做，ADR 0006 的 P0 也不用重新核准。工程交付、實驗覆蓋、模型選型、部署授權分開追蹤；零 creation 不能宣稱 adaptive 有效，0/30 不能宣稱真實誤認率為零。下一份 implementation plan 必須先處理本機 camera/UI 可行性及敏感研究資料保存驗證，再經 operator 實作 go。文件 merge 不啟動相機、不蒐集真人資料。
