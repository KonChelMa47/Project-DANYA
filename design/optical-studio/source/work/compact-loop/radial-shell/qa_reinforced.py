import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];v=np.fromfile(p/'bevel-reinforced.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/'bevel-reinforced.indices.bin','<u4').reshape(-1,3);N=len(v)//2;outer=f[(f<N).all(1)];ov=np.fromfile(root/'work/adult-kit/original-head.positions.bin','<f4').reshape(-1,3);of=np.fromfile(root/'work/adult-kit/head-outer-original.indices.bin','<u4').reshape(-1,3);edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),1);edges,c=np.unique(edges,axis=0,return_counts=True);parent=np.arange(len(v))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
for a,b in edges:
 a=find(a);b=find(b)
 if a!=b:parent[b]=a
_,components=np.unique([find(i) for i in range(len(v))],return_counts=True);t=v[f].astype(float);area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
r={'exteriorTrianglesExactlyPreserved':bool(np.array_equal(v[outer],ov[of])),'openEdges':int((c==1).sum()),'nonManifoldEdges':int((c>2).sum()),'connectedComponentVertexCounts':sorted(components.tolist(),reverse=True),'degenerateTriangles':int((area<1e-10).sum()),'signedVolumeMM3':float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6),'scope':'Topology, exterior coordinate preservation and signed volume; does not prove absence of internal self-intersections or a global minimum thickness.'};(p/'reinforced-qa.json').write_text(json.dumps(r,indent=2));print(r)

roots=np.array([find(i) for i in range(len(v))]);values,counts=np.unique(roots,return_counts=True);details=[]
for value,count in zip(values,counts):
 if count>100:continue
 ids=np.flatnonzero(roots==value);ff=f[roots[f[:,0]]==value];tri=v[ff].astype(float);details.append({'vertices':int(count),'bounds':[v[ids].min(0).tolist(),v[ids].max(0).tolist()],'volumeMM3':float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6),'vertexIndices':ids.tolist()})
(p/'small-components.json').write_text(json.dumps(details,indent=2));print(details)
