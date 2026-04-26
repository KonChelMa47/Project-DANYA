#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/runtime/logs"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SERVER_FILE="${SERVER_FILE:-$ROOT_DIR/apps/tts/gpt_sovits_server.py}"
RESTART_INTERVAL_SEC="${RESTART_INTERVAL_SEC:-3600}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-5}"
HEALTHCHECK_INTERVAL_SEC="${HEALTHCHECK_INTERVAL_SEC:-2}"
STOP_TIMEOUT_SEC="${STOP_TIMEOUT_SEC:-15}"

server_pid=""
server_pgid=""
stopping=0

stop_server() {
  if [[ -z "${server_pid:-}" ]]; then
    return
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    return
  fi

  if [[ -n "${server_pgid:-}" ]]; then
    echo "[SUPERVISOR] stopping process group $server_pgid"
    kill -TERM "-$server_pgid" 2>/dev/null || true
  else
    echo "[SUPERVISOR] stopping pid $server_pid"
    kill -TERM "$server_pid" 2>/dev/null || true
  fi

  for _ in $(seq 1 "$STOP_TIMEOUT_SEC"); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" 2>/dev/null || true
      return
    fi
    sleep 1
  done

  if [[ -n "${server_pgid:-}" ]]; then
    echo "[SUPERVISOR] forcing process group $server_pgid"
    kill -KILL "-$server_pgid" 2>/dev/null || true
  else
    echo "[SUPERVISOR] forcing pid $server_pid"
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
}

shutdown() {
  stopping=1
  stop_server
  exit 0
}

trap shutdown INT TERM HUP
trap '[[ "$stopping" -eq 1 ]] || stop_server' EXIT

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

  setsid bash -c 'cd "$1" || exit 1; exec "$2" "$3"' _ "$ROOT_DIR" "$PYTHON_BIN" "$SERVER_FILE" >>"$log_file" 2>&1 &
  server_pid=$!
  server_pgid="$server_pid"
  launched_epoch="$(date +%s)"

  while kill -0 "$server_pid" 2>/dev/null; do
    now_epoch="$(date +%s)"
    elapsed="$((now_epoch - launched_epoch))"
    if (( elapsed >= RESTART_INTERVAL_SEC )); then
      echo "[SUPERVISOR] scheduled restart after ${elapsed}s (pid=$server_pid)"
      stop_server
      break
    fi
    sleep "$HEALTHCHECK_INTERVAL_SEC"
  done

  wait "$server_pid" 2>/dev/null
  exit_code=$?
  server_pid=""
  server_pgid=""
  stopped_at="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[SUPERVISOR] server stopped at $stopped_at with exit_code=$exit_code"
  echo "[SUPERVISOR] restarting in ${RESTART_DELAY_SEC}s"
  sleep "$RESTART_DELAY_SEC"
done
