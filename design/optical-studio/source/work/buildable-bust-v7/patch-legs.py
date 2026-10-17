import json
from pathlib import Path
source=open('work/buildable-bust-v6/build.py').read();exec(source.split('# Horn bolts:')[0])
out=Path('outputs/danya-simulator/assets/buildable-bust-v7');parts=json.loads((out/'parts.json').read_text())['parts'];parts=[p for p in parts if not p['name'].startswith(('yaw-upper-leg','pitch-bearing-cap'))]
s=open('work/buildable-bust-v7/build.py').read();exec(s[s.index('# Upper yoke'):s.index('# Standalone')]);(out/'parts.json').write_text(json.dumps({'status':'v7 CAD candidate; proof pending','parts':parts},indent=2))
