import numpy as np,json,zipfile
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');z=zipfile.ZipFile(r/'complete-latest/DANYA-prototype-ALL-PARTS-seam-v1.zip');dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);rep={}
for n in z.namelist():
 if not n.startswith('01_PRINT/')or not n.endswith('.stl'):continue
 raw=np.frombuffer(z.read(n),dtype=dt,offset=84)['v'];v,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3);e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);u,c=np.unique(e,axis=0,return_counts=True);a=np.linalg.norm(np.cross(raw[:,1].astype(float)-raw[:,0],raw[:,2].astype(float)-raw[:,0]),axis=1);rep[n]={'nonManifoldEdges':int((c!=2).sum()),'zeroAreaFaces':int((a<1e-10).sum())};print(n,rep[n],flush=True)
Path('work/assembly-integrity/zip-geometry.json').write_text(json.dumps(rep,indent=2))
