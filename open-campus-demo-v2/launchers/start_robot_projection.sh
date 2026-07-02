#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export PYGLET_SHADOW_WINDOW="${PYGLET_SHADOW_WINDOW:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

echo "[ROBOT PROJECTION] starting pyglet projection window"
cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/open-campus-demo-v2/robot_projection/robot_projection.py" "$@"
