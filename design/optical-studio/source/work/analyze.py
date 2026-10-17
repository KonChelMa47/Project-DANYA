import numpy as np, json
from pathlib import Path
for p in Path(__file__).parent.glob('DANYA*.npz'):
 d=np.load(p); v=d['v']; f=d['loops'].reshape(-1,3)
 e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
 ue,c=np.unique(e,axis=0,return_counts=True)
 be=ue[c==1]; bv=np.unique(be)
 print(p.stem, json.dumps({'vertices':len(v),'faces':len(f),'boundary_edges':len(be),'nonmanifold_edges':int(sum(c>2)),'boundary_bounds': [v[bv].min(0).tolist(),v[bv].max(0).tolist()] if len(bv) else None,'edge_lengths':np.percentile(np.linalg.norm(v[ue[:,0]]-v[ue[:,1]],axis=1),[0,10,50,90,99,100]).tolist()}),flush=True)
 np.savez(p.with_name(p.stem+'-topology.npz'),edges=ue,boundary=be)
