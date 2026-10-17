import numpy as np,json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');d=np.load(r/'thick-head-v3/thickness-samples.npz');v=d['v'];n=d['norm'];C=np.array([-22,0,165]);scale=np.array([85,80,115]);ok=(d['before']>.85)&(d['before']<1.25)&(v[:,2]>52)&(v[:,0]<25);v=v[ok][::4];n=n[ok][::4];p=v+n*3.3;u=(p-C)/scale;rad=np.linalg.norm(u,axis=1);u/=rad[:,None];powers=[(i,j,k)for i in range(11)for j in range(11-i)for k in range(min(1,10-i-j)+1)]
def mat(u):return np.stack([u[:,0]**i*u[:,1]**j*u[:,2]**k for i,j,k in powers],1)
A=mat(u);w=np.ones(len(p))
for _ in range(7):
 coef=np.linalg.solve((A.T*w)@A+np.eye(len(powers))*1e-6,(A.T*w)@rad);err=(A@coef-rad);w=1/np.maximum(1,abs(err)/.005)
print('FIT MM error',np.percentile(abs(err)*85,[0,50,90,99,100]));np.savez(r/'smooth-head-v4/fit.npz',coef=coef,powers=powers,C=C,scale=scale)
