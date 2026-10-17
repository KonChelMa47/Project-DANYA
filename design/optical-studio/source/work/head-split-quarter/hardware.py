import numpy as np,json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'head-split-quarter';a=json.loads((r/'split-head-v6/head-split-check.json').read_text());b=json.loads((out/'head-split-check.json').read_text());dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);rows=[]
for i in range(4):
 t=np.array(b['flanges'][i]['boltCenter'])-np.array(a['flanges'][i]['boltCenter'])
 for typ in ['bolt','nut','washer']:
  n=f'head-seam-{typ}-{i}';d=np.fromfile(r/'assembly'/(n+'.stl'),dtype=dt,offset=84);d['v']+=t; (out/(n+'.stl')).write_bytes(b'DANYA purchased hardware envelope'.ljust(80,b' ')+np.uint32(len(d)).tobytes()+d.tobytes());rows.append(n)
for i in range(2):
 n=f'head-alignment-pin-{i}';t=np.array(b['pins'][i]['center'])-np.array(a['pins'][i]['center']);d=np.fromfile(r/'assembly'/(n+'.stl'),dtype=dt,offset=84);d['v']+=t;(out/(n+'.stl')).write_bytes(b'DANYA alignment pin'.ljust(80,b' ')+np.uint32(len(d)).tobytes()+d.tobytes());rows.append(n)
(out/'hardware.json').write_text(json.dumps({'parts':[{'name':n}for n in rows]},indent=2))
