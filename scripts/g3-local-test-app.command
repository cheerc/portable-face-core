#!/bin/sh
# G3 本機辨識測試 App 啟動器（W7, spec §2-1：Finder 點兩下即開）。
#
# 前提（結論見 PR body「前提1」）：
# - canonical checkout：本機 ~/portable-face-core（可經 FACECORE_REPO 覆蓋）；
#   啟動時嘗試 git pull --ff-only 更新到 main，失敗則用現有 checkout 繼續。
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
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git pull --ff-only origin main >/dev/null 2>&1; then
    echo "提醒：main 更新失敗（離線或非 ff-only），用現有 checkout 繼續。"
  fi
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
uv run --no-sync --extra research-ui python -m facecore.research.cli live \
  --profile profiles/g3-v1.json \
  --store "$HOME/Downloads/face_sample/_facecore/store" \
  --device 0 \
  --session "$SESSION" \
  --record-consent \
  --image-consent \
  --ui qt \
  --continuous \
  --config "$CONFIG"
