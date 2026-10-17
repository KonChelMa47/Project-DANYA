import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;v=np.fromfile(p.parent/'adult-kit/original-face.positions.bin','<f4').reshape(-1,3).astype(float);f=np.fromfile(p.parents[1]/'outputs/danya-simulator/assets/face.indices.bin','<u4').reshape(-1,3);of=np.fromfile(p.parent/'adult-kit/original-face-outer.indices.bin','<u4');mask=np.ones(len(v),bool);mask[np.unique(of)]=False
edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);a=np.r_[edges[:,0],edges[:,1]];b=np.r_[edges[:,1],edges[:,0]];cnt=np.bincount(a,minlength=len(v));w=v.copy()
w=np.fromfile(p/'face-smoothed-80.positions.bin','<f4').reshape(-1,3).astype(float)
center=np.array([14.085,-62.874,107.804]);weight=np.maximum(0,1-np.linalg.norm(v-center,axis=1)/4);weight[~mask]=0
for iteration in range(1,21):
 sums=np.zeros_like(v)
 for j in range(3):sums[:,j]=np.bincount(a,weights=w[b,j],minlength=len(v))
 w+=.5*(sums/np.maximum(cnt,1)[:,None]-w)*weight[:,None]
name='face-local';w.astype('<f4').tofile(p/(name+'.positions.bin'));f.tofile(p/(name+'.indices.bin'));print('maximum interior change',np.linalg.norm(w-v,axis=1).max())
