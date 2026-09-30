#!/bin/sh
# G3 本機辨識測試 App 啟動器（W7, spec §2-1：Finder 點兩下即開）。
#
# 前提（結論見 PR body「前提1」）：
# - canonical checkout：本機 ~/portable-face-core（可經 FACECORE_REPO 覆蓋）。
#   D4（decision d-20260929235132588526-14）：啟動時**不再** git pull。
#   固定版本驗收要求「用哪一版測試」由 operator 決定並記錄，若 launcher
#   自行快轉，驗收結果不可重現。改為顯示目前 commit 與是否落後
#   origin/main；落後時**只提示、不動作**，要更新由 operator 自己決定。
# - 環境：uv sync --extra research-ui（含 opencv-python-headless 真機擷取）。
# - 設定：~/Downloads/face_sample/_facecore/g3-local.json（範本見 repo
#   docs/g3-local-config.example.json）；缺失則中文提示並停住。
# - 相機：視窗內下拉選單由 operator 選擇；首次 macOS 相機權限彈窗歸在
#   Terminal 名下（.command 經 Terminal 執行），詳 SOP。
set -u

REPO="${FACECORE_REPO:-$HOME/portable-face-core}"
CONFIG="$HOME/Downloads/face_sample/_facecore/g3-local.json"

if [ ! -d "$REPO" ]; then
  echo "找不到 canonical checkout：$REPO"
  echo "請先 git clone 到該位置，或設 FACECORE_REPO 再重試。"
  read -r -p "按 Enter 關閉… " _dummy
  exit 2
fi

cd "$REPO" || exit 2

# D4（decision d-20260929235132588526-14）: 顯示版本，不改版本。
#
# 這一段**不 pull、不 checkout、不 reset、不 stash、不 merge**。working tree
# 與 HEAD 保持 operator 啟動時的狀態 —— D4 要的是「operator 決定測哪一版」，
# launcher 自行快轉會讓驗收不可重現（D0 §11 第 5、12 項）。
#
# `git fetch` 只更新遠端追蹤 ref origin/main，不動 working tree、不動 HEAD、
# 不動本機 branch，所以**不算修改 repo 狀態**；它存在的唯一理由是：拿本機
# 可能過期的 origin/main 去比對並宣稱「已是最新」是一個對 operator 的假宣稱，
# 而 D4 的驗收記錄正依賴這個數字。fetch 失敗時不阻擋啟動（App 照跑），
# 改為明說「無法確認」並標示依本機快取比對 —— 寧可少講一句，不可講錯一句。
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  CURRENT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
  CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  FETCH_OK=1
  git fetch --quiet origin main >/dev/null 2>&1 || FETCH_OK=0

  echo "G3 本機測試 App — 目前版本"
  echo "  分支：$CURRENT_BRANCH"
  echo "  commit：$CURRENT_SHA"
  if [ "$FETCH_OK" -eq 1 ]; then
    BEHIND="$(git rev-list --count HEAD..origin/main 2>/dev/null || echo '?')"
    if [ "$BEHIND" = "0" ]; then
      echo "  狀態：與 origin/main 一致"
    else
      echo "  狀態：落後 origin/main $BEHIND 個 commit（**不自動更新**）"
      echo "  要更新請自己在 Terminal 執行 git -C \"$REPO\" pull --ff-only origin main"
    fi
  else
    echo "  狀態：無法連線確認是否落後（git fetch 失敗，以下依本機快取的 origin/main 比對）"
    BEHIND_LOCAL="$(git rev-list --count HEAD..origin/main 2>/dev/null || echo '?')"
    if [ "$BEHIND_LOCAL" = "0" ]; then
      # 刻意不用「一致」：fetch 沒成功時，這個 repo 有沒有落後是未知的，
      # 拿可能過期的本機 ref 講「一致」是對 operator 的假宣稱。
      echo "  依本機快取：顯示無落後（**本機快取可能過期，未經連線確認**）"
    else
      echo "  依本機快取：顯示落後 $BEHIND_LOCAL 個 commit（本機快取可能過期）"
    fi
  fi
  echo "  請把這個 commit 記進 D4 驗收記錄。"
  echo
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "找不到 uv，請先安裝 uv（https://docs.astral.sh/uv/）再重試。"
  read -r -p "按 Enter 關閉… " _dummy
  exit 2
fi

if ! uv sync --extra research-ui; then
  echo "環境備妥失敗（uv sync），請檢查網路後重試。"
  read -r -p "按 Enter 關閉… " _dummy
  exit 2
fi

if [ ! -f "$CONFIG" ]; then
  echo "找不到本機設定檔：$CONFIG"
  echo "請依 repo docs/g3-local-config.example.json 複製填寫後重試。"
  read -r -p "按 Enter 關閉… " _dummy
  exit 2
fi

SESSION="g3-$(date +%Y%m%d-%H%M%S)"
# D3b (decision d-20260929193651396921-13): the double-clicked App is the
# NON-RECORDING demo path. It writes no encrypted frames, no embeddings and
# no attempt ledger; each labeled round lands as one plaintext row in
# demo-results.csv beside results.csv. Consent flags are deliberately absent
# — demo mode refuses them, because accepting them here would wrongly imply
# that recording is on. Use `--mode record --record-consent --image-consent`
# for the research executor.
uv run --no-sync --extra research-ui python -m facecore.research.cli live \
  --profile profiles/g3-v1.json \
  --store "$HOME/Downloads/face_sample/_facecore/store" \
  --device 0 \
  --session "$SESSION" \
  --ui qt \
  --continuous \
  --mode demo \
  --config "$CONFIG"
