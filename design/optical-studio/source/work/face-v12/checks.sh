#!/bin/zsh
nodebin=/Users/daniilmalchenko/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node
for name in magnet-carrier-covered head-upper-covered head-lower-covered magnet-clear-0-face-magnet magnet-clear-0-head-magnet magnet-clear-1-face-magnet magnet-clear-1-head-magnet; do
 "$nodebin" work/face-v12/rays.mjs "$name"
done
for name in magnet-carrier-covered head-upper-covered head-lower-covered; do
 "$nodebin" work/face-v12/surface.mjs "$name"
done
