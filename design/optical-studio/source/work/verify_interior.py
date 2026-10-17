import bpy,numpy as np,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
bpy.ops.wm.open_mainfile(filepath=str(root/'work/interior_candidate.blend'))
current={n:bpy.data.objects[n] for n in ['DANYA-face-back','DANYA-face-shell']}
with bpy.data.libraries.load(str(root/'work/before_interior.blend'),link=False) as (s,d):d.objects=list(current)
def arrays(m):
 v=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get('co',v);m.calc_loop_triangles();f=np.empty(len(m.loop_triangles)*3,dtype=np.int32);m.loop_triangles.foreach_get('vertices',f)
 return v.reshape(-1,3),f.reshape(-1,3)
ref={o.name.rsplit('.',1)[0]:(o,arrays(o.data)) for o in d.objects}
report={}
fit=np.load(root/'work/seam_fit.npz')
def cap(v,f):
 yz=(v[:,1:]-[0,135])/100;x=sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
 on=abs(v[:,0]-x)<.00001;tri=v[f[np.all(on[f],axis=1)]]
 return {tuple(sorted(tuple(p) for p in t)) for t in tri}
for name,o in current.items():
 v,f=arrays(o.data);rv,rf=ref[name][1]
 if name.endswith('shell'):
  report['face_geometry_unchanged']=bool(np.array_equal(v,rv) and np.array_equal(f,rf));assert report['face_geometry_unchanged']
 else:
  beforecap=cap(rv,rf);aftercap=cap(v,f);report['interface_triangles_unchanged']=len(beforecap.symmetric_difference(aftercap))==0;report['interface_triangles']=len(aftercap);assert report['interface_triangles_unchanged']
  center=v[f].mean(1);dd=.866025403784*center[:,0]-.5*center[:,2]+90.8
  panel=(abs(dd)<2.5)&(abs(center[:,1])<35)&(center[:,2]>112)&(center[:,2]<205)
  report['remaining_triangles_in_panel_center']=int(sum(panel));assert not np.any(panel)
  e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);ue,c=np.unique(e,axis=0,return_counts=True)
  area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
  report['boundary_edges']=int(sum(c==1));report['nonmanifold_edges']=int(sum(c>2));report['degenerate_triangles']=int(sum(area<1e-10))
  assert report['boundary_edges']==report['nonmanifold_edges']==report['degenerate_triangles']==0
  parent=np.arange(len(v),dtype=np.int32)
  for _ in range(100):
   low=np.minimum(parent[ue[:,0]],parent[ue[:,1]]);np.minimum.at(parent,ue[:,0],low);np.minimum.at(parent,ue[:,1],low)
   for _ in range(12):parent=parent[parent]
   if np.all(parent[ue[:,0]]==parent[ue[:,1]]):break
  report['back_connected_components']=len(np.unique(parent));assert report['back_connected_components']==1
report['polish']=json.loads((root/'work/interior_polish.json').read_text())
(root/'work/interior_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
