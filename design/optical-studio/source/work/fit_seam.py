import numpy as np, json
from mathutils.kdtree import KDTree
from pathlib import Path
root=Path(__file__).parent
a=np.load(root/'DANYA-face-back.npz')['v']; b=np.load(root/'DANYA-face-shell.npz')['v']
kd=KDTree(len(a))
for i,v in enumerate(a): kd.insert(v,i)
kd.balance()
p=[]; dist=[]
for v in b[::3]:
 co,i,d=kd.find(v)
 if d<1.3: p.append((v+a[i])/2); dist.append(d)
p=np.array(p); np.save(root/'seam_points.npy',p)
print('seam points',len(p),'bounds',p.min(0),p.max(0),flush=True)
for degree in [1,2,3,4,5]:
 yz=(p[:,1:]-[0,135])/100
 powers=[(i,j) for i in range(degree+1) for j in range(degree+1-i)]
 A=np.array([yz[:,0]**i*yz[:,1]**j for i,j in powers]).T
 w=np.ones(len(p))
 for _ in range(8):
  c=np.linalg.lstsq(A*w[:,None],p[:,0]*w,rcond=None)[0]; r=A@c-p[:,0]; w=1/np.maximum(1,np.abs(r)/1.0)
 print(degree,'residual',np.percentile(np.abs(r),[50,90,95,99,100]).tolist(),flush=True)
 if degree==3: np.savez(root/'seam_fit.npz',coeff=c,powers=np.array(powers))
print('done',flush=True)
