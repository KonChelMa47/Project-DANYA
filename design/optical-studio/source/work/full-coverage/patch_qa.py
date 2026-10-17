import numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;result={};meshes={}
for name in ['head','patch']:
 v=np.fromfile(p/f'{name}.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/f'{name}.indices.bin','<u4').reshape(-1,3);meshes[name]=(v,f)
 edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);e,n=np.unique(edges,axis=0,return_counts=True)
 parent=np.arange(len(v));
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for a,b in e:
  a=find(a);b=find(b)
  if a!=b:parent[b]=a
 roots=np.array([find(i) for i in range(len(v))]);comp,cnt=np.unique(roots,return_counts=True)
 tri=v[f].astype(np.float64);area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 result[name]={'vertices':len(v),'triangles':len(f),'bounds':[v.min(axis=0).tolist(),v.max(axis=0).tolist()],'boundaryEdges':int((n==1).sum()),'nonManifoldEdges':int((n>2).sum()),'components':sorted(cnt.tolist(),reverse=True),'minTriangleArea':float(area.min()),'tinyTriangles':int((area<1e-8).sum()),'signedVolume':float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)}
a={tuple(x) for x in meshes['head'][0]};result['exactSharedVertices']=sum(tuple(x)in a for x in meshes['patch'][0]);print(json.dumps(result,indent=2));(p/'patch-qa.json').write_text(json.dumps(result,indent=2))
