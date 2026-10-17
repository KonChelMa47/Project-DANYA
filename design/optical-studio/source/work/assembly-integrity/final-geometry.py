import numpy as np,json,hashlib
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype/solid-seam-v2');dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);rep={}
for name in ['head-upper','head-lower','magnet-carrier']:
 p=r/(name+'.stl');raw=np.fromfile(p,dtype=dt,offset=84)['v'];v,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3);e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);u,c=np.unique(e,axis=0,return_counts=True);areas=np.linalg.norm(np.cross(raw[:,1].astype(float)-raw[:,0],raw[:,2].astype(float)-raw[:,0]),axis=1);rep[name]={'triangles':len(f),'nonManifoldEdges':int((c!=2).sum()),'degenerateTriangles':int((areas<1e-10).sum()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bounds':[v.min(0).tolist(),v.max(0).tolist()]};print(name,rep[name],flush=True)
(r/'geometry-validation.json').write_text(json.dumps(rep,indent=2))
