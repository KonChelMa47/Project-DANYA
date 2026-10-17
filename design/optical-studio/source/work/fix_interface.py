import bpy, bmesh, numpy as np, time
from pathlib import Path
root=Path(__file__).resolve().parent.parent; t=time.time()
def log(s): print(f'[{time.time()-t:.1f}s] {s}',flush=True)
fit=np.load(root/'work/seam_fit.npz')
def seam(v):
 yz=(v[:,1:]-[0,135])/100
 return sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
def coords(m):
 a=np.empty(len(m.vertices)*3,dtype=np.float32); m.vertices.foreach_get('co',a); return a.reshape(-1,3)
bpy.ops.wm.open_mainfile(filepath=str(root/'work/candidate.blend'))
face=bpy.data.objects['DANYA-face-shell']; back=bpy.data.objects['DANYA-face-back']
for o in [face,back]:
 v=coords(o.data); v[:,0]-=seam(v); on=np.abs(v[:,0])<.0001; v[on,0]=0
 o.data.vertices.foreach_set('co',v.ravel()); o.data.update()
bm=bmesh.new(); bm.from_mesh(face.data)
cap=[f for f in bm.faces if all(abs(v.co.x)<.0001 for v in f.verts)]
bmesh.ops.triangulate(bm,faces=cap)
bm.to_mesh(face.data); bm.free(); face.data.update()
cap=[p for p in face.data.polygons if all(abs(face.data.vertices[i].co.x)<.0001 for i in p.vertices)]
log(f'Triangulated interface in flat space: {len(cap)} faces')
bm=bmesh.new(); bm.from_mesh(back.data)
oldcap=[f for f in bm.faces if all(abs(v.co.x)<.0001 for v in f.verts)]
bmesh.ops.delete(bm,geom=oldcap,context='FACES')
shared={}
for p in cap:
 verts=[]
 for vi in reversed(p.vertices):
  if vi not in shared: shared[vi]=bm.verts.new(face.data.vertices[vi].co)
  verts.append(shared[vi])
 bm.faces.new(verts)
rim=[v for v in bm.verts if abs(v.co.x)<.0001]
bmesh.ops.remove_doubles(bm,verts=rim,dist=.00005)
bm.to_mesh(back.data); bm.free(); back.data.update()
for o in [face,back]:
 m=o.data; v=coords(m); on=np.abs(v[:,0])<.0001
 for p in m.polygons: p.use_smooth=not all(on[i] for i in p.vertices)
 v[:,0]+=seam(v); m.vertices.foreach_set('co',v.ravel()); m.update()
 bm=bmesh.new(); bm.from_mesh(m); bmesh.ops.recalc_face_normals(bm,faces=bm.faces); bm.to_mesh(m); bm.free(); m.update()
 m.calc_loop_triangles(); f=np.empty(len(m.loop_triangles)*3,dtype=np.int32); m.loop_triangles.foreach_get('vertices',f)
 np.savez(root/'work'/f'{o.name}-repaired.npz',v=coords(m),f=f.reshape(-1,3))
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/candidate.blend')); log('Saved second candidate')
