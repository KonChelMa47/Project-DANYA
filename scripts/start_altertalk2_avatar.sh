#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
ALTERTALK_URL="${ALTERTALK_URL:-http://127.0.0.1:8765}"
BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${BRIDGE_PORT:-8767}"
BRIDGE_PORT_MAX="${BRIDGE_PORT_MAX:-8799}"
BRIDGE_POLL_SEC="${BRIDGE_POLL_SEC:-1}"
START_ALTERTALK="${START_ALTERTALK:-1}"
ALTERTALK_MODE="${ALTERTALK_MODE:-full}"
SHOW_TRACKER="${SHOW_TRACKER:-0}"
TRACKER_TERMINAL_ONLY="${TRACKER_TERMINAL_ONLY:-0}"
TRACKER_MOCK_VISION="${TRACKER_MOCK_VISION:-0}"
TRACKER_CAMERA_CANDIDATES="${TRACKER_CAMERA_CANDIDATES:-}"
TRACKER_CAMERA_INDEX="${TRACKER_CAMERA_INDEX:-}"
AVATAR_EXTRA_ARGS="${AVATAR_EXTRA_ARGS:-}"
ALTERTALK_PID=""
BRIDGE_PID=""

usage() {
  cat <<'EOF'
Usage: scripts/start_altertalk2_avatar.sh [--agent-only] [--no-altertalk] [avatar args...]

Starts:
  1. Open Campus Demo v1 speech generator on http://127.0.0.1:8765
  2. Open Campus Demo v1 -> /api/output bridge on http://127.0.0.1:8767
  3. apps/avatar/conversation_avatar.py connected to that bridge

Options:
  --agent-only      Start only conversation_agent, without tracker
  --no-altertalk    Reuse an already running speech server
  --show-tracker    Show tracker logs, useful when the camera does not open
  --terminal-only   Run tracker without an OpenCV display window
  --mock-vision     Run tracker without OpenAI image analysis
  --camera-index N   Prefer one tracker camera index
  --cameras A,B      Try tracker camera indexes in this order
  -h, --help        Show this help

Environment:
  ALTERTALK_URL       Source speech server URL (default: http://127.0.0.1:8765)
  BRIDGE_PORT         Bridge /api/output port (default: 8767)
  BRIDGE_PORT_MAX     Highest bridge port to try (default: 8799)
  BRIDGE_POLL_SEC     Bridge poll interval (default: 1)
  SHOW_TRACKER=1      Same as --show-tracker
  TRACKER_TERMINAL_ONLY=1
  TRACKER_MOCK_VISION=1
  TRACKER_CAMERA_INDEX=2
  TRACKER_CAMERA_CANDIDATES=2,3
  AVATAR_EXTRA_ARGS   Extra args passed to apps/avatar/conversation_avatar.py
EOF
}

cleanup() {
  if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "[ALTERTALK AVATAR] stopping bridge"
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
  if [[ -n "$ALTERTALK_PID" ]] && kill -0 "$ALTERTALK_PID" 2>/dev/null; then
    echo "[ALTERTALK AVATAR] stopping Open Campus Demo v1"
    kill "$ALTERTALK_PID" 2>/dev/null || true
    wait "$ALTERTALK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

check_json_health() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1.5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    raise SystemExit(0 if payload.get("ok") is True else 1)
except Exception:
    raise SystemExit(1)
PY
}

check_bridge_health() {
  "$PYTHON_BIN" - "$1" "$ALTERTALK_URL" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

url = sys.argv[1]
source = sys.argv[2].rstrip("/")
try:
    with urllib.request.urlopen(url, timeout=1.5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("ok") is not True:
        raise SystemExit(1)
    source_url = str(payload.get("source_url") or "").rstrip("/")
    raise SystemExit(0 if source_url == source else 1)
except Exception:
    raise SystemExit(1)
PY
}

check_text_health() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1.5) as response:
        body = response.read().decode("utf-8")
    raise SystemExit(0 if body.strip() else 1)
except Exception:
    raise SystemExit(1)
PY
}

port_is_free() {
  "$PYTHON_BIN" - "$BRIDGE_HOST" "$1" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
raise SystemExit(0)
PY
}

choose_bridge_port() {
  local port="$BRIDGE_PORT"
  while (( port <= BRIDGE_PORT_MAX )); do
    local base_url="http://$BRIDGE_HOST:$port"
    if check_bridge_health "$base_url/api/health"; then
      echo "$port:healthy"
      return 0
    fi
    if port_is_free "$port"; then
      echo "$port:free"
      return 0
    fi
    echo "[ALTERTALK AVATAR] bridge port $port is in use; trying next port" >&2
    port=$((port + 1))
  done
  return 1
}

wait_for_url() {
  local url="$1"
  local mode="$2"
  local label="$3"
  local tries="${4:-30}"

  for ((i = 1; i <= tries; i++)); do
    if [[ "$mode" == "json" ]]; then
      if check_json_health "$url"; then
        return 0
      fi
    else
      if check_text_health "$url"; then
        return 0
      fi
    fi
    sleep 1
  done

  echo "[ALTERTALK AVATAR] timed out waiting for $label: $url" >&2
  return 1
}

AVATAR_CLI_ARGS=()
while (($# > 0)); do
  case "$1" in
    --agent-only)
      ALTERTALK_MODE="agent-only"
      ;;
    --no-altertalk)
      START_ALTERTALK=0
      ;;
    --show-tracker)
      SHOW_TRACKER=1
      ;;
    --terminal-only|--tracker-terminal-only)
      TRACKER_TERMINAL_ONLY=1
      ;;
    --mock-vision|--mock-vlm|--tracker-mock-vision)
      TRACKER_MOCK_VISION=1
      ;;
    --camera-index|--tracker-camera-index)
      if (($# < 2)); then
        echo "[ALTERTALK AVATAR] missing value for $1" >&2
        exit 1
      fi
      TRACKER_CAMERA_INDEX="$2"
      shift
      ;;
    --cameras|--camera-candidates|--tracker-camera-candidates)
      if (($# < 2)); then
        echo "[ALTERTALK AVATAR] missing value for $1" >&2
        exit 1
      fi
      TRACKER_CAMERA_CANDIDATES="$2"
      shift
      ;;
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
  echo "[ALTERTALK AVATAR] Python not found or not executable: $PYTHON_BIN" >&2
  echo "[ALTERTALK AVATAR] Create the venv first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

if [[ "$START_ALTERTALK" != "0" ]]; then
  if [[ "$ALTERTALK_MODE" == "agent-only" ]]; then
    echo "[ALTERTALK AVATAR] starting Open Campus Demo v1 agent only"
    (
      cd "$ROOT_DIR/open-campus-demo-v1/services/conversation_agent"
      export DANYA_IDLE_SHORT_MIN_SEC="${DANYA_IDLE_SHORT_MIN_SEC:-3}"
      export DANYA_IDLE_SHORT_MAX_SEC="${DANYA_IDLE_SHORT_MAX_SEC:-8}"
      export DANYA_IDLE_LONG_INTERVAL_SEC="${DANYA_IDLE_LONG_INTERVAL_SEC:-30}"
      "$PYTHON_BIN" main.py
    ) &
  else
    altertalk_args=()
    if [[ "$SHOW_TRACKER" == "1" ]]; then
      altertalk_args+=(--show-tracker)
    fi
    if [[ "$TRACKER_TERMINAL_ONLY" == "1" ]]; then
      altertalk_args+=(--tracker-terminal-only)
    fi
    if [[ "$TRACKER_MOCK_VISION" == "1" ]]; then
      altertalk_args+=(--tracker-mock-vision)
    fi
    echo "[ALTERTALK AVATAR] starting Open Campus Demo v1 tracker + agent"
    (
      cd "$ROOT_DIR/open-campus-demo-v1"
      if [[ -n "$TRACKER_CAMERA_INDEX" ]]; then
        export DANYA_TRACKER_CAMERA_INDEX="$TRACKER_CAMERA_INDEX"
      fi
      if [[ -n "$TRACKER_CAMERA_CANDIDATES" ]]; then
        export DANYA_TRACKER_CAMERA_CANDIDATES="$TRACKER_CAMERA_CANDIDATES"
      fi
      "$PYTHON_BIN" launchers/run_open_campus_demo.py "${altertalk_args[@]}"
    ) &
  fi
  ALTERTALK_PID=$!
fi

wait_for_url "$ALTERTALK_URL/health" text "Open Campus Demo v1 speech server" 45

if ! selected_bridge="$(choose_bridge_port)"; then
  echo "[ALTERTALK AVATAR] no usable bridge port found in range $BRIDGE_PORT-$BRIDGE_PORT_MAX" >&2
  exit 1
fi

BRIDGE_PORT="${selected_bridge%%:*}"
BRIDGE_PORT_STATE="${selected_bridge##*:}"

if [[ "$BRIDGE_PORT_STATE" == "healthy" ]]; then
  echo "[ALTERTALK AVATAR] existing bridge is healthy: http://$BRIDGE_HOST:$BRIDGE_PORT"
else
  echo "[ALTERTALK AVATAR] starting bridge on http://$BRIDGE_HOST:$BRIDGE_PORT"
  "$PYTHON_BIN" "$ROOT_DIR/apps/demo_servers/open_campus_speech_bridge.py" \
    --host "$BRIDGE_HOST" \
    --port "$BRIDGE_PORT" \
    --altertalk-url "$ALTERTALK_URL" \
    --poll-sec "$BRIDGE_POLL_SEC" &
  BRIDGE_PID=$!
fi

wait_for_url "http://$BRIDGE_HOST:$BRIDGE_PORT/api/health" json "Open Campus Demo v1 bridge" 15

export DANYA_LLM_OUTPUT_SERVER="http://$BRIDGE_HOST:$BRIDGE_PORT"
export DANYA_LLM_OUTPUT_INTERVAL="${DANYA_LLM_OUTPUT_INTERVAL:-1}"
export DANYA_CONTROL_PORT="${DANYA_CONTROL_PORT:-8766}"

echo "[ALTERTALK AVATAR] starting DANYA conversation avatar"
cd "$ROOT_DIR"

if [[ -n "$AVATAR_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$ROOT_DIR/apps/avatar/conversation_avatar.py" $AVATAR_EXTRA_ARGS "${AVATAR_CLI_ARGS[@]}"
else
  "$PYTHON_BIN" "$ROOT_DIR/apps/avatar/conversation_avatar.py" "${AVATAR_CLI_ARGS[@]}"
fi
