import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;backup=p/'head-before-bevel.positions.bin'
if not backup.exists():backup.write_bytes((p/'head.positions.bin').read_bytes())
v=np.fromfile(backup,'<f4').reshape(-1,3).copy();pv=np.fromfile(p/'patch.positions.bin','<f4').reshape(-1,3);outer={tuple(x) for x in pv[:len(pv)//2]};lock=np.array([tuple(x) in outer for x in v]);s=json.loads((p/'design.json').read_text());origin=np.array(s['virtual']);poly=np.array([(s['coneReferenceX'],y,z) for y,z in s['cone']]);norm=np.cross(poly-origin,np.roll(poly,-1,axis=0)-origin);norm/=np.linalg.norm(norm,axis=1)[:,None];center=poly.mean(0);norm[np.einsum('ij,j->i',norm,center-origin)>0]*=-1
# Interior clearance bevel: preserve the exact external contact contour, relieve the inner joint.
dist=(v-origin)@norm.T;plane=dist.argmax(1);distance=dist.max(1);weight=np.clip(1-np.abs(distance)/3,0,1);weight=weight**2*(3-2*weight);weight[lock]=0;w=v+.005*weight[:,None]*norm[plane];w.astype('<f4').tofile(p/'head.positions.bin');print('Moved vertices',int((weight>0).sum()),'Locked contact vertices',int(lock.sum()),'max move',float(np.max(np.linalg.norm(w-v,axis=1))));s['innerJointReliefMM']=.005;(p/'design.json').write_text(json.dumps(s,indent=2))
