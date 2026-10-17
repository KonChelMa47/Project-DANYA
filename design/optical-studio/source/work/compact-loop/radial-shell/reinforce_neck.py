import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;v=np.fromfile(p/'bevel-0.6.positions.bin','<f4').reshape(-1,3).astype(float);f=np.fromfile(p/'bevel-0.6.indices.bin','<u4').reshape(-1,3);N=len(v)//2;outer=f[(f<N).all(1)];n=np.zeros((N,3));c=np.cross(v[outer[:,1]]-v[outer[:,0]],v[outer[:,2]]-v[outer[:,0]])
for j in range(3):np.add.at(n,outer[:,j],c)
n/=np.linalg.norm(n,axis=1)[:,None];th=-((v[N:]-v[:N])*n).sum(1);correction=np.maximum(0,1-th);correction[th>.999]=0;v[N:]-=n*correction[:,None];v.astype('<f4').tofile(p/'bevel-reinforced.positions.bin');f.tofile(p/'bevel-reinforced.indices.bin');r={'changedInnerVertices':int((correction>0).sum()),'maximumCorrectionMM':float(correction.max()),'scope':'Increase normal-projected separation at capped neck vertices to nominal 1 mm; exact exterior unchanged; actual minimum wall thickness still requires geometric validation.'};(p/'reinforcement.json').write_text(json.dumps(r,indent=2));print(r)
