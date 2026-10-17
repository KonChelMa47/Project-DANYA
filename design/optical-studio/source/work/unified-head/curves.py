import numpy as np,json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');d=np.load(r/'face-v8/screen-data.npz');print('SCREEN',[(k,d[k].shape)for k in d]);v=d['after'];f=d['outerFaces'];edges=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);e,c=np.unique(edges,axis=0,return_counts=True);adj={}
for a,b in e[c==1]:adj.setdefault(int(a),[]).append(int(b));adj.setdefault(int(b),[]).append(int(a))
ids=[next(iter(adj))];prev=None
while True:
 nxt=next(n for n in adj[ids[-1]]if n!=prev)
 if nxt==ids[0]:break
 prev=ids[-1];ids.append(nxt)
pts=v[ids];ds=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1);cum=np.r_[0,np.cumsum(ds)];N=720;front=np.array([np.interp(np.arange(N)*cum[-1]/N,cum,np.r_[pts[:,k],pts[0,k]])for k in range(3)]).T
old=np.load(r/'printable-seam/band-samples.npz');outside=old['carrierOutside'];inside=old['carrierInside'];print('OLD front distance',np.percentile(np.linalg.norm(outside[0]-front,axis=1),[0,50,90,100]))
def smooth(p,k=26):
 ft=np.fft.rfft(p,axis=0);freq=np.arange(len(ft));ft*=np.exp(-(freq/k)**6)[:,None];return np.fft.irfft(ft,n=N,axis=0)
rear=smooth(outside[-1]);frontOriginal=front.copy();front=smooth(outside[0]);S=41;oldSmooth=np.stack([smooth(row)for row in outside]);outer=np.stack([np.array([np.interp(t,np.linspace(0,1,9),oldSmooth[:,i,k])for i in range(N)for k in range(3)]).reshape(N,3)for t in np.linspace(0,1,S)]);inner=np.stack([np.array([np.interp(t,np.linspace(0,1,9),inside[:,i,k])for i in range(N)for k in range(3)]).reshape(N,3)for t in np.linspace(0,1,S)]);np.savez(r/'unified-head-v5/band-samples.npz',carrierOutside=outer,carrierInside=inner,faceBoundaryOriginal=frontOriginal,faceBoundary=front+np.array([.12,0,0]));print('newfront rise',np.percentile(np.linalg.norm(front-frontOriginal,axis=1),[0,50,90,100]))
