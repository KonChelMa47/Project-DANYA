import numpy as np, json, sys
from pathlib import Path
root=Path(__file__).parent
results={}
for path in [root/p for p in sys.argv[1:]]:
 d=np.load(path); v=d['v']; f=d['f']; e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
 ue,c=np.unique(e,axis=0,return_counts=True)
 cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
 area=np.linalg.norm(cross,axis=1)/2
 volume=np.einsum('ij,ij->i',v[f[:,0]].astype('float64'),cross.astype('float64')).sum()/6
 result={'vertices':len(v),'triangles':len(f),'boundary_edges':int(sum(c==1)),'nonmanifold_edges':int(sum(c>2)),'degenerate_triangles':int(sum(area<1e-10)),'signed_volume':float(volume),'surface_area':float(area.sum()),'bounds':[v.min(0).tolist(),v.max(0).tolist()]}
 results[path.stem]=result
 print(path.name,json.dumps(result),flush=True)
(root/'mesh_checks.json').write_text(json.dumps(results,indent=2))
