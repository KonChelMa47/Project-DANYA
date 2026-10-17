import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;v=np.fromfile(p/'head-before-bevel.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/'head.indices.bin','<u4').reshape(-1,3);bv=np.fromfile(p/'baseline.positions.bin','<f4').reshape(-1,3);keys={tuple(x) for x in bv};shared=np.array([tuple(x) in keys for x in v]);keep=f[~shared[f].all(1)]
edges=np.concatenate([keep[:,[0,1]],keep[:,[1,2]],keep[:,[2,0]]]);_,ii,cc=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True);boundary=edges[ii[cc==1]];adj={}
for a,b in boundary:adj.setdefault(int(a),[]).append(int(b));adj.setdefault(int(b),[]).append(int(a))
print('Removed',len(f)-len(keep),'boundary',len(boundary),'baddegree',sum(len(x)!=2 for x in adj.values()),flush=True)
loops=[];seen=set()
for a in adj:
 if a in seen:continue
 loop=[];current=a;last=-1
 while current not in seen:
  seen.add(current);loop.append(current);nxt=adj[current][0] if adj[current][0]!=last else adj[current][1];last,current=current,nxt
 loops.append(loop)
print('loops',list(map(len,loops)),flush=True)
if len(loops)!=2:raise RuntimeError('Expected two join loops')
loops.sort(key=lambda l:float(v[l,0].mean()),reverse=True);outer,inner=[np.array(x) for x in loops];pv=np.fromfile(p/'patch.positions.bin','<f4').reshape(-1,3);facekeys={tuple(x) for x in pv[:len(pv)//2]};print('Outer matched',sum(tuple(x) in facekeys for x in v[outer]),len(outer),flush=True)
# Unroll the optical cone by polar angle, then zipper corresponding inner / outer edges.
for i,loop in enumerate(loops):
 angle=np.arctan2(v[loop,2]-135,v[loop,1]);order=np.argsort(angle);loops[i]=np.array(loop)[order]
a,b=loops;ta=np.arctan2(v[a,2]-135,v[a,1]);tb=np.arctan2(v[b,2]-135,v[b,1]);i=j=0;tri=[]
while i<len(a) or j<len(b):
 an=ta[(i+1)%len(a)]+(2*np.pi if i+1>=len(a) else 0);bn=tb[(j+1)%len(b)]+(2*np.pi if j+1>=len(b) else 0)
 if j>=len(b) or (i<len(a) and an<bn):tri.append([a[i%len(a)],a[(i+1)%len(a)],b[j%len(b)]]);i+=1
 else:tri.append([a[i%len(a)],b[(j+1)%len(b)],b[j%len(b)]]);j+=1
newf=np.concatenate([keep,np.array(tri)]).astype('<u4');v.tofile(p/'head.positions.bin');newf.tofile(p/'joint-head.indices.bin');print('Joint triangles',len(tri),flush=True)
