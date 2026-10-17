import bpy, numpy as np, json, time
from pathlib import Path
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree
root=Path(__file__).resolve().parent.parent; start=time.time()
def log(s): print(f'[{time.time()-start:.1f}s] {s}',flush=True)
fit=np.load(root/'work/seam_fit.npz')
def seam(v):
 yz=(v[:,1:]-[0,135])/100
 return sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
bpy.ops.wm.open_mainfile(filepath=str(root/'work/candidate.blend'))
report={}; data={}; caps={}
for kind in ['back','shell']:
 name='DANYA-face-'+kind; d=np.load(root/'work'/f'{name}-repaired.npz'); v=d['v']; f=d['f']; data[kind]=(v,f)
 on=np.abs(v[:,0]-seam(v))<.00001; cap=f[np.all(on[f],axis=1)]; caps[kind]=(v[on],v[cap])
 # Shared-edge face connectivity, including all tiny isolated fragments.
 o=bpy.data.objects[name]; m=o.data
 edges=np.empty(len(m.edges)*2,dtype=np.int32); m.edges.foreach_get('vertices',edges); edges=edges.reshape(-1,2)
 parent=np.arange(len(v),dtype=np.int32)
 # Parallel label propagation with path compression converges quickly on triangle meshes.
 for it in range(100):
  lo=np.minimum(parent[edges[:,0]],parent[edges[:,1]])
  np.minimum.at(parent,edges[:,0],lo); np.minimum.at(parent,edges[:,1],lo)
  for _ in range(12): parent=parent[parent]
  if np.all(parent[edges[:,0]]==parent[edges[:,1]]): break
 _,counts=np.unique(parent,return_counts=True)
 log(f'{kind} components: {sorted(counts.tolist(),reverse=True)[:15]}')
 # Quantify preserved original surfaces away from the intentionally rebuilt rim.
 tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
 original=np.load(root/'work'/f'{name}.npz')['v']; original=original[::max(1,len(original)//18000)]
 away=np.abs(original[:,0]-seam(original))>8
 deviations=np.array([tree.find_nearest(p)[3] for p in original[away]])
 report[kind]={'component_sizes':sorted(counts.tolist(),reverse=True),'interface_vertices':int(sum(on)),'interface_triangles':len(cap),'original_surface_distance_away_from_seam':{'samples':len(deviations),'median':float(np.median(deviations)),'p95':float(np.percentile(deviations,95)),'max':float(deviations.max())}}
 log(f'{kind} deviation {report[kind]}')
 del tree
a,ta=caps['back']; b,tb=caps['shell']; kd=KDTree(len(a))
for i,p in enumerate(a): kd.insert(p,i)
kd.balance(); dist=np.array([kd.find(p)[2] for p in b]); report['interface']={'max_vertex_gap_model_units':float(dist.max()),'rms_vertex_gap_model_units':float(np.sqrt(np.mean(dist**2)))}
def keys(tris):
 return {tuple(sorted(tuple(np.round(p,5)) for p in tri)) for tri in tris}
ka,kb=keys(ta),keys(tb); report['interface']['unmatched_triangles']=len(ka.symmetric_difference(kb)); report['interface']['matching_triangles']=len(ka&kb)
log(f'Interface {report["interface"]}')
(root/'work/verification.json').write_text(json.dumps(report,indent=2)); log('Verification saved')
