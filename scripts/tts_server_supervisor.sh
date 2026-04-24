#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/runtime/logs"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SERVER_FILE="${SERVER_FILE:-$ROOT_DIR/server.py}"
RESTART_INTERVAL_SEC="${RESTART_INTERVAL_SEC:-3600}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-5}"
HEALTHCHECK_INTERVAL_SEC="${HEALTHCHECK_INTERVAL_SEC:-2}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[SUPERVISOR] Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$SERVER_FILE" ]]; then
  echo "[SUPERVISOR] Server file not found: $SERVER_FILE" >&2
  exit 1
fi

echo "[SUPERVISOR] root=$ROOT_DIR"
echo "[SUPERVISOR] python=$PYTHON_BIN"
echo "[SUPERVISOR] server=$SERVER_FILE"
echo "[SUPERVISOR] restart_interval=${RESTART_INTERVAL_SEC}s"
echo "[SUPERVISOR] restart_delay=${RESTART_DELAY_SEC}s"

while true; do
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  stamp="$(date '+%Y%m%d_%H%M%S')"
  log_file="$LOG_DIR/tts_server_${stamp}.log"

  echo "[SUPERVISOR] starting server at $started_at"
  echo "[SUPERVISOR] log_file=$log_file"

  (
    cd "$ROOT_DIR" || exit 1
    exec "$PYTHON_BIN" "$SERVER_FILE"
  ) >>"$log_file" 2>&1 &
  server_pid=$!
  launched_epoch="$(date +%s)"

  while kill -0 "$server_pid" 2>/dev/null; do
    now_epoch="$(date +%s)"
    elapsed="$((now_epoch - launched_epoch))"
    if (( elapsed >= RESTART_INTERVAL_SEC )); then
      echo "[SUPERVISOR] scheduled restart after ${elapsed}s (pid=$server_pid)"
      kill -TERM "$server_pid" 2>/dev/null || true
      sleep 10
      if kill -0 "$server_pid" 2>/dev/null; then
        echo "[SUPERVISOR] forcing shutdown for pid=$server_pid"
        kill -KILL "$server_pid" 2>/dev/null || true
      fi
      break
    fi
    sleep "$HEALTHCHECK_INTERVAL_SEC"
  done

  wait "$server_pid"
  exit_code=$?
  stopped_at="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[SUPERVISOR] server stopped at $stopped_at with exit_code=$exit_code"
  echo "[SUPERVISOR] restarting in ${RESTART_DELAY_SEC}s"
  sleep "$RESTART_DELAY_SEC"
done
