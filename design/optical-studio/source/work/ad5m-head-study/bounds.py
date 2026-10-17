import numpy as np,json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype/assembly')
v=np.concatenate([np.fromfile(r/(n+'.stl'),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'].reshape(-1,3)for n in ['head-upper','head-lower']]);v=np.unique(v,axis=0)
# Rigid orientations only: examine face-opening roughly downward (original +X downward).
base=np.column_stack((v[:,1],v[:,2],-v[:,0]));rows=[]
for deg in range(0,91):
 a=np.deg2rad(deg);c,s=np.cos(a),np.sin(a);q=base.copy();q[:,:2]=base[:,:2]@np.array([[c,s],[-s,c]]);dims=np.ptp(q,axis=0);rows.append({'angle':deg,'dimensions':dims.tolist(),'maxXY':float(max(dims[:2]))})
rows.sort(key=lambda x:x['maxXY']);print('Whole-head rigid orientation candidates',rows[:3]);Path('work/ad5m-head-study/orientation-bounds.json').write_text(json.dumps(rows,indent=2))
