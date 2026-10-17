import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;v=np.fromfile(p.parent/'adult-kit/original-face.positions.bin','<f4').reshape(-1,3).astype(float);f=np.fromfile(p.parents[1]/'outputs/danya-simulator/assets/face.indices.bin','<u4').reshape(-1,3);of=np.fromfile(p.parent/'adult-kit/original-face-outer.indices.bin','<u4');mask=np.ones(len(v),bool);mask[np.unique(of)]=False
edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);a=np.r_[edges[:,0],edges[:,1]];b=np.r_[edges[:,1],edges[:,0]];cnt=np.bincount(a,minlength=len(v));w=v.copy()
for iteration in range(1,321):
 sums=np.zeros_like(v)
 for j in range(3):sums[:,j]=np.bincount(a,weights=w[b,j],minlength=len(v))
 proposed=w+.5*(sums/np.maximum(cnt,1)[:,None]-w);delta=proposed-v;delta*=np.minimum(1,2.5/np.maximum(np.linalg.norm(delta,axis=1),1e-10))[:,None];w[mask]=(v+delta)[mask]
 if iteration in [20,40,80,160,320]:
  name=f'face-smoothed-{iteration}';w.astype('<f4').tofile(p/(name+'.positions.bin'));f.tofile(p/(name+'.indices.bin'));print(name,float(np.linalg.norm(w-v,axis=1).max()),flush=True)
