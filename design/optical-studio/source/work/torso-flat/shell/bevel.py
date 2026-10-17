import numpy as np,json,heapq
from pathlib import Path
p=Path(__file__).resolve().parent;name='optical-t1.0';v=np.fromfile(p/f'{name}.positions.bin','<f4').reshape(-1,3).astype(float);f=np.fromfile(p/f'{name}.indices.bin','<u4').reshape(-1,3);N=len(v)//2;outer=f[(f<N).all(axis=1)];n=np.zeros((N,3));cross=np.cross(v[outer[:,1]]-v[outer[:,0]],v[outer[:,2]]-v[outer[:,0]])
for j in range(3):np.add.at(n,outer[:,j],cross)
n/=np.linalg.norm(n,axis=1)[:,None];e=np.sort(np.concatenate([outer[:,[0,1]],outer[:,[1,2]],outer[:,[2,0]]]),axis=1);edges,cnt=np.unique(e,axis=0,return_counts=True);boundary=np.unique(edges[cnt==1]);adj=[[] for _ in range(N)]
for a,b in edges:
 dist=np.linalg.norm(v[a]-v[b]);adj[a].append((b,dist));adj[b].append((a,dist))
distances=np.full(N,np.inf);queue=[]
for i in boundary:distances[i]=0;heapq.heappush(queue,(0,i))
while queue:
 d,i=heapq.heappop(queue)
 if d>distances[i]:continue
 for j,w in adj[i]:
  nd=d+w
  if nd<4 and nd<distances[j]:distances[j]=nd;heapq.heappush(queue,(nd,j))
tangent=np.tile([-1.,0,0],(N,1));tangent-=n*(tangent*n).sum(1)[:,None];tangent/=np.maximum(1e-8,np.linalg.norm(tangent,axis=1))[:,None];weight=np.maximum(0,1-distances/4)
for amount in [.15,.3,.6,1.0]:
 vv=v.copy();vv[N:]+=tangent*(amount*weight)[:,None];name=f'bevel-{amount}';vv.astype('<f4').tofile(p/f'{name}.positions.bin');f.tofile(p/f'{name}.indices.bin');print(name)
