#!/bin/zsh
pybin=/Users/daniilmalchenko/Documents/Project-DANYA/.venv/bin/python
nodebin=/Users/daniilmalchenko/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node
for part in magnet-carrier-covered head-upper-covered head-lower-covered; do
 "$pybin" work/face-v12/meshfix.py "$part" "$part"
done
"$nodebin" work/face-v12/magnet-fit.mjs
