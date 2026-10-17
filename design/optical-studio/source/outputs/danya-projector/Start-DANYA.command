#!/bin/zsh
cd "$(dirname "$0")"
if curl -fs --max-time 1 http://127.0.0.1:8771/api/status >/dev/null; then
  open http://127.0.0.1:8771/
else
  python3 server.py
fi
