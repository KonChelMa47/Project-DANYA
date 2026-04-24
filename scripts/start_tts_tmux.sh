#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-danya_tts}"
SUPERVISOR_SCRIPT="$ROOT_DIR/scripts/tts_server_supervisor.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

if [[ ! -x "$SUPERVISOR_SCRIPT" ]]; then
  chmod +x "$SUPERVISOR_SCRIPT"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" "$SUPERVISOR_SCRIPT"
echo "started tmux session: $SESSION_NAME"
