import json,hashlib
from pathlib import Path
r=Path('outputs/danya-simulator/assets/head-connectors-v3');d=json.load(open(r/'motion-check.json'));unexpected=[]
for p in d['changedParts']:
 n=p['name'];allowed=set()
 if n.startswith('trial-'):
  side='left'if'left'in n else'right';ids=[0,2]if side=='left'else[1,3];allowed={'head','roll-carrier-v7','root-left-upper-washer'if side=='left'else'root-head-right-upper-washer',*[f'head-connect-{i}-washer-2'for i in ids]}
 elif n=='head':allowed={'face','trial-left-head-rail','trial-right-head-rail','printed-mirror-fixed-plate',*[f'mirror-head-{kind}-{i}'for kind in ['bolt','washer']for i in range(2)],*[f'magnet-clear-{i}-head-magnet'for i in range(4)]}
 else:
  idx=int(n.split('-')[2]);allowed={f'head-connect-{idx}-{suffix}'for suffix in ['bolt','nut','washer-0','washer-1','washer-2']};allowed.add('trial-'+('left'if idx%2==0 else'right')+'-head-rail')
 p['unexplainedSurfaceContacts']=[x for x in p['hits']if x not in allowed];unexpected.extend([[n,x]for x in p['unexplainedSurfaceContacts']])
d['unexpectedContacts']=unexpected;d['classification']='Declared matching faces, washer stacks and simplified unthreaded screw/nut envelopes excluded. Changed head/support mating volumes checked separately; this does not establish continuous clearance.';d['inputSHA256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in r.glob('*.stl')if not p.name.startswith('head-neck')};(r/'motion-check.json').write_text(json.dumps(d,indent=2));print('UNEXPECTED',unexpected);assert not unexpected
