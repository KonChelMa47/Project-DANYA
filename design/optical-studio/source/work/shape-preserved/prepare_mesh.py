from pathlib import Path
import numpy as np,json
p=Path(__file__).resolve().parent
root=p.parent.parent/'outputs/danya-simulator'
s={'k':1,'sy':1}; arrays=[];faces=[];off=0
for name in ['head','face']:
 v=np.fromfile(root/'assets'/f'{name}.positions.bin',dtype='<f4').reshape(-1,3).copy();f=np.fromfile(root/'assets'/f'{name}.indices.bin',dtype='<u4').reshape(-1,3)
 if name=='head':
  t=np.clip(-v[:,0]/25,0,1);w=t*t*(3-2*t);v[:,0]*=1+(s['k']-1)*w;v[:,1]*=1+(s['sy']-1)*w
 arrays.append(v);faces.append(f+off);off+=len(v)
v,inv=np.unique(np.concatenate(arrays),axis=0,return_inverse=True);f=inv[np.concatenate(faces)]
_,index,counts=np.unique(np.sort(f,axis=1),axis=0,return_inverse=True,return_counts=True);internal=counts[index]>1
print('shared internal triangles removed',int(internal.sum()),flush=True)
assert internal.sum()==8826
f=f[~internal];edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);_,ec=np.unique(edges,axis=0,return_counts=True)
assert np.all(ec==2)
np.savez(p/'unified.npz',v=v.astype('<f4'),f=f.astype('<i4'))
print('unified',len(v),len(f),v.min(0),v.max(0),flush=True)
