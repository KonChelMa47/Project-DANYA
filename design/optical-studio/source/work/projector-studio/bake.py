import numpy as np,json
from pathlib import Path
root=Path('outputs/danya-simulator/assets/projection-prototype/unified-head-v5')
b=(root/'face.stl').read_bytes(); n=int.from_bytes(b[80:84],'little'); dt=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]);t=np.frombuffer(b,dt,n,84)['v'].astype(float)
uv=np.fromfile(root/'face-preview/uv.bin',dtype='<f4').reshape(-1,3,2);lit=np.fromfile(root/'face-preview/lit.bin',dtype='<f4').reshape(-1,3)
front=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])[:,0]>0
lo=t[front].reshape(-1,3).min(0);hi=t[front].reshape(-1,3).max(0)
src=np.stack(((t[:,:,1]-lo[1])/(hi[1]-lo[1]),1-(t[:,:,2]-lo[2])/(hi[2]-lo[2])),axis=-1)
out=np.full((720,1280,2),65535,dtype='<u2');count=0
for p,q in zip(uv[front&(lit>.5).all(1)],src[front&(lit>.5).all(1)]):
 p=p.copy();p[:,0]=p[:,0]*1280-.5;p[:,1]=(1-p[:,1])*720-.5
 x0,y0=np.maximum(np.ceil(p.min(0)).astype(int),[0,0]);x1,y1=np.minimum(np.floor(p.max(0)).astype(int),[1279,719])
 if x1<x0 or y1<y0:continue
 a,b,c=p;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
 if abs(den)<1e-10:continue
 yy,xx=np.mgrid[y0:y1+1,x0:x1+1];u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den;v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;w=1-u-v;mask=(u>=-1e-5)&(v>=-1e-5)&(w>=-1e-5)
 vals=u[...,None]*q[0]+v[...,None]*q[1]+w[...,None]*q[2];out[y0:y1+1,x0:x1+1][mask]=np.round(np.clip(vals[mask],0,1)*65534).astype('<u2')
out.tofile('outputs/danya-projector/assets/head-v5.uv16');print('valid pixels',int((out[:,:,0]!=65535).sum()))
