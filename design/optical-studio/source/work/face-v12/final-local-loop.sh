#!/bin/zsh
set -e
for iteration in 1 2 3; do
 /Applications/Blender.app/Contents/MacOS/Blender -b -t 4 --python work/face-v12/local-repair-blender.py > work/face-v12/local-loop-$iteration-blender.log 2>&1
 /Users/daniilmalchenko/Documents/Project-DANYA/.venv/bin/python work/face-v12/meshfix-rounded.py magnet-carrier-covered magnet-carrier-covered > work/face-v12/local-loop-$iteration-meshfix.log 2>&1
 /Users/daniilmalchenko/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node work/face-v12/surface.mjs magnet-carrier-covered > work/face-v12/local-loop-$iteration-surface.log 2>&1
 if python3 -c "import json;d=json.load(open('outputs/danya-simulator/assets/projection-prototype/face-v12/magnet-carrier-covered-surface.json'));print(d['crossingsOver01MM'],d['coplanarOverlapsOver00001MM2']);exit(bool(d['crossingsOver01MM']+d['coplanarOverlapsOver00001MM2']))"; then break; fi
done
