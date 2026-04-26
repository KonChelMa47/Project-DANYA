#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8767}"
LLM_PORT_MAX="${LLM_PORT_MAX:-8799}"
LLM_INTERVAL="${LLM_INTERVAL:-20}"
LLM_IMMEDIATE="${LLM_IMMEDIATE:-0}"
START_LLM_SERVER="${START_LLM_SERVER:-1}"
AVATAR_EXTRA_ARGS="${AVATAR_EXTRA_ARGS:-}"
LLM_STARTED=0
AVATAR_CLI_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/start_conversation_demo.sh [--with-llm|--without-llm] [avatar args...]

Options:
  --with-llm       Start or reuse the local demo LLM output server (default)
  --without-llm    Start only the conversation avatar; disable LLM output polling
  --no-llm         Alias for --without-llm
  -h, --help       Show this help

Environment:
  START_LLM_SERVER=0  Same as --without-llm
  LLM_IMMEDIATE=1     Emit one demo LLM output immediately
  AVATAR_EXTRA_ARGS   Extra args passed to apps/avatar/conversation_avatar.py
EOF
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_falsey() {
  case "${1,,}" in
    0|false|no|off) return 0 ;;
    *) return 1 ;;
  esac
}

while (($# > 0)); do
  case "$1" in
    --with-llm)
      START_LLM_SERVER=1
      ;;
    --without-llm|--no-llm|--no-llm-server)
      START_LLM_SERVER=0
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
  echo "[DEMO] Python not found or not executable: $PYTHON_BIN" >&2
  echo "[DEMO] Create the venv first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

cleanup() {
  if [[ "$LLM_STARTED" == "1" && -n "${LLM_PID:-}" ]] && kill -0 "$LLM_PID" 2>/dev/null; then
    echo "[DEMO] stopping LLM demo server"
    kill "$LLM_PID" 2>/dev/null || true
    wait "$LLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

check_llm_health() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=1.5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    raise SystemExit(0 if payload.get("ok") is True else 1)
except Exception:
    raise SystemExit(1)
PY
}

port_is_free() {
  "$PYTHON_BIN" - "$LLM_HOST" "$1" <<'PY' >/dev/null 2>&1
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

choose_llm_port() {
  local port="$LLM_PORT"
  while (( port <= LLM_PORT_MAX )); do
    local base_url="http://$LLM_HOST:$port"
    if check_llm_health "$base_url/api/health"; then
      echo "$port:healthy"
      return 0
    fi
    if port_is_free "$port"; then
      echo "$port:free"
      return 0
    fi
    echo "[DEMO] port $port is in use but not a healthy DANYA LLM server; trying next port" >&2
    port=$((port + 1))
  done
  return 1
}

if is_falsey "$START_LLM_SERVER"; then
  echo "[DEMO] starting without LLM demo server"
  AVATAR_CLI_ARGS=(--no-llm-output "${AVATAR_CLI_ARGS[@]}")
else
  if ! is_truthy "$START_LLM_SERVER"; then
    echo "[DEMO] invalid START_LLM_SERVER value: $START_LLM_SERVER" >&2
    echo "[DEMO] Use 1/true/yes/on or 0/false/no/off." >&2
    exit 1
  fi

  if ! selected="$(choose_llm_port)"; then
    echo "[DEMO] no usable LLM port found in range $LLM_PORT-$LLM_PORT_MAX" >&2
    exit 1
  fi

  LLM_PORT="${selected%%:*}"
  LLM_PORT_STATE="${selected##*:}"
  llm_args=(
    "$ROOT_DIR/apps/demo_servers/llm_output_demo_server.py"
    --host "$LLM_HOST"
    --port "$LLM_PORT"
    --interval "$LLM_INTERVAL"
  )

  if is_truthy "$LLM_IMMEDIATE"; then
    llm_args+=(--immediate)
  fi

  LLM_BASE_URL="http://$LLM_HOST:$LLM_PORT"
  LLM_HEALTH_URL="$LLM_BASE_URL/api/health"

  if [[ "$LLM_PORT_STATE" == "healthy" ]]; then
    echo "[DEMO] existing LLM server is healthy: $LLM_BASE_URL"
  else
    echo "[DEMO] starting LLM demo server on $LLM_BASE_URL"
    "$PYTHON_BIN" "${llm_args[@]}" &
    LLM_PID=$!
    LLM_STARTED=1

    sleep 1

    if ! kill -0 "$LLM_PID" 2>/dev/null; then
      wait "$LLM_PID" 2>/dev/null || true
      echo "[DEMO] LLM demo server exited immediately. Port $LLM_PORT may already be in use." >&2
      echo "[DEMO] Check the process using: ss -ltnp 'sport = :$LLM_PORT'" >&2
      exit 1
    fi

    if ! check_llm_health "$LLM_HEALTH_URL"; then
      echo "[DEMO] LLM demo server started but health check failed: $LLM_HEALTH_URL" >&2
      exit 1
    fi
  fi

  export DANYA_LLM_OUTPUT_SERVER="$LLM_BASE_URL"
  export DANYA_LLM_OUTPUT_INTERVAL="$LLM_INTERVAL"
fi

echo "[DEMO] starting DANYA conversation avatar"
cd "$ROOT_DIR"

if [[ -n "$AVATAR_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$ROOT_DIR/apps/avatar/conversation_avatar.py" $AVATAR_EXTRA_ARGS "${AVATAR_CLI_ARGS[@]}"
else
  "$PYTHON_BIN" "$ROOT_DIR/apps/avatar/conversation_avatar.py" "${AVATAR_CLI_ARGS[@]}"
fi
