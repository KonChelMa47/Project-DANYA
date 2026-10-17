import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent
v=np.fromfile(p/'baseline.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/'outer.indices.bin','<u4').reshape(-1,3)
used,inv=np.unique(f,return_inverse=True);v=v[used];f=inv.reshape(-1,3);n=len(v)
edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);keys=np.sort(edges,axis=1);_,idx,cnt=np.unique(keys,axis=0,return_index=True,return_counts=True);border=edges[idx[cnt==1]]
virtual=np.array(json.loads((p/'design.json').read_text())['virtual']);inner=virtual+(v-virtual)*.996
newv=np.concatenate([v,inner]).astype('<f4');newf=np.concatenate([f,f[:,::-1]+n,np.array([[a,b+n,b] for a,b in border]),np.array([[a,a+n,b+n] for a,b in border])]).astype('<u4')
a={tuple(x) for x in np.fromfile(p/'head.positions.bin','<f4').reshape(-1,3)};bv=np.unique(border);print({'outerVertices':n,'triangles':len(f),'boundaryEdges':len(border),'sharedBoundary':sum(tuple(v[x])in a for x in bv),'boundaryVertices':len(bv),'nonmanifold':int((cnt>2).sum())},flush=True)
newv.tofile(p/'patch.positions.bin');newf.tofile(p/'patch.indices.bin')
