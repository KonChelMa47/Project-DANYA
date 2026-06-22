#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
AVATAR_EXTRA_ARGS="${AVATAR_EXTRA_ARGS:-}"

usage() {
  cat <<'EOF'
Usage: scripts/start_low_power_avatar.sh [avatar args...]

Starts only the DANYA still avatar and its control web endpoint.
YOLO, camera access, recorded motion auto-play, idle blinking, and LLM output polling are disabled.

Environment:
  AVATAR_EXTRA_ARGS   Extra args passed to apps/avatar/conversation_avatar.py
  PYTHON_BIN          Python executable to use
EOF
}

AVATAR_CLI_ARGS=()
while (($# > 0)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      AVATAR_CLI_ARGS+=("$@")
      break
      ;;
    *)
      AVATAR_CLI_ARGS+=("$1")
      ;;
  esac
  shift
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[LOW POWER] Python not found or not executable: $PYTHON_BIN" >&2
  echo "[LOW POWER] Create the venv first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

echo "[LOW POWER] starting still DANYA avatar without YOLO/camera/LLM"
echo "[LOW POWER] control web will be available if the avatar stays open"
cd "$ROOT_DIR"

if [[ -n "$AVATAR_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$ROOT_DIR/apps/avatar/conversation_avatar.py" --static --windowed $AVATAR_EXTRA_ARGS "${AVATAR_CLI_ARGS[@]}"
else
  "$PYTHON_BIN" "$ROOT_DIR/apps/avatar/conversation_avatar.py" --static --windowed "${AVATAR_CLI_ARGS[@]}"
fi
