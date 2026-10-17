import json
from pathlib import Path
root=Path('outputs/danya-simulator/assets/buildable-bust-v10');f=root/'trial-structural-candidate.json';d=json.load(open(f))
for manifest,remove in [('trial-pitch-support.json',['pitch-outboard-bridge','frame-pitch-motor-support'])]:
 if (root/manifest).exists():
  rows=json.load(open(root/manifest))['parts'];names=[p['name']for p in rows]+remove;d['parts']=[p for p in d['parts']if p['name']not in names]+rows
p=next(p for p in json.load(open(root.with_name('buildable-bust-v8')/'parts.json'))['parts']if p['name']=='roll-idler-carrier-inner');d['parts']=[r for r in d['parts']if r['name']!=p['name']]+[p];f.write_text(json.dumps(d,indent=2))
