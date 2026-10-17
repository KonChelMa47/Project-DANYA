import numpy as np,json,sys
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype/unified-head-v5');dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);data=np.fromfile(r/(sys.argv[1]+'.stl'),dtype=dt,offset=84);raw=data['v'];v,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3);area=np.linalg.norm(np.cross(raw[:,1].astype(float)-raw[:,0],raw[:,2].astype(float)-raw[:,0]),axis=1);print('degen',sum(area<1e-10))
parent=np.arange(len(v))
for tri in f[area<1e-10]:
 pairs=[(a,b,float(np.linalg.norm(v[a].astype(float)-v[b])))for a,b in zip(tri,np.roll(tri,1))];a,b,d=min(pairs,key=lambda p:p[2]);
 if d<1e-4:parent[b]=a
for _ in range(5):parent=parent[parent]
f=parent[f];keep=(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,2]!=f[:,0])&(area>=1e-10);f=f[keep];_,unique=np.unique(np.sort(f,axis=1),axis=0,return_index=True);f=f[np.sort(unique)];raw=v[f];d=np.zeros(len(f),dtype=dt);d['v']=raw;nn=np.cross(raw[:,1]-raw[:,0],raw[:,2]-raw[:,0]);nn/=np.maximum(np.linalg.norm(nn,axis=1)[:,None],1e-30);d['n']=nn;(r/(sys.argv[1]+'-preclean.stl')).write_bytes(b'DANYA remove submicron print transform degeneracies'.ljust(80,b' ')+np.uint32(len(d)).tobytes()+d.tobytes());e,c=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0,return_counts=True);print('RESULT',len(f),int((c!=2).sum()))
