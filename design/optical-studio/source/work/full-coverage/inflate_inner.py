import numpy as np,sys
from pathlib import Path
p=Path(__file__).resolve().parent
d=np.load(p/'face-before-polish.npz');v=d['v'].copy();f=d['f'];mask=np.fromfile(p/'face-inner-mask.bin','u1')>0
n=np.zeros_like(v);tri=v[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);
for j in range(3):np.add.at(n,f[:,j],norm)
n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
# Move only internal walls into the material, enlarging the optical cavity.
w=v.copy();amount=float(sys.argv[1]);weight=np.clip((45-v[:,0])/8,0,1);weight=weight*weight*(3-2*weight)
w[mask]-=amount*n[mask]*weight[mask,None]
w.astype('<f4').tofile(p/'face.positions.bin');print('Max move',np.linalg.norm(w-v,axis=1).max())
