import bpy, numpy as np, time, json, bmesh
from pathlib import Path
root=Path(__file__).resolve().parent.parent; t=time.time()
def log(s): print(f'[{time.time()-t:.1f}s] {s}',flush=True)
def coords(m):
 a=np.empty(len(m.vertices)*3,dtype=np.float32); m.vertices.foreach_get('co',a); return a.reshape(-1,3)
def apply(o,m):
 bpy.context.view_layer.objects.active=o; bpy.ops.object.modifier_apply(modifier=m.name)
 log(f'{o.name}: {len(o.data.vertices)} vertices, {len(o.data.polygons)} faces')
def seam(v):
 yz=(v[:,1:]-[0,135])/100
 return sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
bpy.ops.wm.open_mainfile(filepath=str(root/('work/common_simplified.blend' if (root/'work/common_simplified.blend').exists() else 'work/union.blend')))
fit=np.load(root/'work/seam_fit.npz')
o=next(o for o in bpy.context.scene.objects if o.type=='MESH')
bpy.context.view_layer.objects.active=o
# Reduce redundancy after repair, before the shared interface is cut.
if not (root/'work/common_simplified.blend').exists():
 m=o.modifiers.new('Efficient smooth surface','DECIMATE'); m.decimate_type='COLLAPSE'; m.ratio=.14; apply(o,m)
 bm=bmesh.new(); bm.from_mesh(o.data); bmesh.ops.triangulate(bm,faces=list(bm.faces)); bm.to_mesh(o.data); bm.free()
 bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/common_simplified.blend'))
log('Reduced common shell')
v=coords(o.data); v[:,0]-=seam(v); o.data.vertices.foreach_set('co',v.ravel()); o.data.update()
back=o; back.name='DANYA-face-back'
face=o.copy(); face.data=o.data.copy(); bpy.context.collection.objects.link(face); face.name='DANYA-face-shell'
bpy.ops.mesh.primitive_cube_add(size=2,location=(250,0,150)); cutter=bpy.context.object; cutter.name='Shared interface cutter'; cutter.scale=(250,200,250)
bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
for obj,operation in [(face,'INTERSECT'),(back,'DIFFERENCE')]:
 m=obj.modifiers.new('Shared zero-clearance interface','BOOLEAN'); m.operation=operation; m.solver='MANIFOLD'; m.object=cutter; apply(obj,m)
bpy.data.objects.remove(cutter,do_unlink=True)
# Force exactly the same interface tessellation in both shells before unwarping.
def cap_faces(m):
 v=coords(m); on=np.abs(v[:,0])<0.0000001
 return [p.index for p in m.polygons if all(on[i] for i in p.vertices)]
cf=cap_faces(face.data); cb=cap_faces(back.data)
# Triangulate the common contact surface before restoring its curved shape.
bm=bmesh.new(); bm.from_mesh(face.data); bm.faces.ensure_lookup_table()
bmesh.ops.triangulate(bm,faces=[bm.faces[i] for i in cf]); bm.to_mesh(face.data); bm.free(); face.data.update()
cf=cap_faces(face.data)
log(f'Interface faces face={len(cf)}, back={len(cb)}')
assert cf and cb
bm=bmesh.new(); bm.from_mesh(back.data); bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm,geom=[bm.faces[i] for i in cb],context='FACES')
shared={}
existing={tuple(v.co):v for v in bm.verts if abs(v.co.x)<.0000001}
for pi in cf:
 p=face.data.polygons[pi]; verts=[]
 for vi in reversed(p.vertices):
  if vi not in shared:
   co=face.data.vertices[vi].co; key=tuple(co)
   shared[vi]=existing[key] if key in existing else bm.verts.new(co)
  verts.append(shared[vi])
 bm.faces.new(verts)
bm.to_mesh(back.data); bm.free(); back.data.update()
for obj in [face,back]:
 m=obj.data; v=coords(m); on=np.abs(v[:,0])<.00006
 vg=obj.vertex_groups.new(name='Exact mating interface'); vg.add(np.where(on)[0].tolist(),1.0,'REPLACE')
 v[:,0]+=seam(v); m.vertices.foreach_set('co',v.ravel()); m.update()
 # Consistent visible outer normals; keep interface caps flat-shaded.
 for p in m.polygons: p.use_smooth=not all(on[i] for i in p.vertices)
 bm=bmesh.new(); bm.from_mesh(m); bmesh.ops.recalc_face_normals(bm,faces=bm.faces); bm.to_mesh(m); bm.free()
 obj['repair_notes']='Scan noise smoothed; topology remeshed; two complementary parts from one common curved interface; zero designed clearance.'
 obj['source_backup']='robot_shell_original.blend'
 obj['model_units_preserved']=True
 m.calc_loop_triangles(); v=coords(m); f=np.empty(len(m.loop_triangles)*3,dtype=np.int32); m.loop_triangles.foreach_get('vertices',f)
 np.savez(root/'work'/f'{obj.name}-repaired.npz',v=v,f=f.reshape(-1,3))
bpy.ops.object.select_all(action='DESELECT'); face.select_set(True); bpy.context.view_layer.objects.active=face
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/candidate.blend'))
log('Candidate saved')
