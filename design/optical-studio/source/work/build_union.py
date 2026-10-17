import bpy, numpy as np, time, json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
t=time.time()
def log(s): print(f'[{time.time()-t:.1f}s] {s}',flush=True)
def arr(mesh,field='co'):
 a=np.empty(len(mesh.vertices)*3,dtype=np.float32); mesh.vertices.foreach_get(field,a); return a.reshape(-1,3)
def apply(o,m):
 bpy.context.view_layer.objects.active=o
 bpy.ops.object.modifier_apply(modifier=m.name)
 log(f'{m.name if False else o.name}: {len(o.data.vertices)} verts {len(o.data.polygons)} faces')
bpy.ops.wm.open_mainfile(filepath=str(root/'outputs/robot_shell_original.blend'))
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
bpy.ops.object.select_all(action='DESELECT')
for o in objects: o.select_set(True)
bpy.context.view_layer.objects.active=objects[0]
bpy.ops.object.join(); o=bpy.context.object; o.name='Unified_work_surface'
log('Joined originals')
# A small outward offset closes scan-sized cracks on remeshing. Reverse after remesh.
m=o.data; v=arr(m); n=arr(m,'normal'); m.vertices.foreach_set('co',(v+0.45*n).ravel()); m.update()
r=o.modifiers.new('Repair topology and close narrow cracks','REMESH'); r.mode='VOXEL'; r.voxel_size=0.32; r.adaptivity=0; r.use_smooth_shade=True
apply(o,r)
m=o.data; v=arr(m); n=arr(m,'normal'); m.vertices.foreach_set('co',(v-0.45*n).ravel()); m.update()
log('Reversed offset')
s=o.modifiers.new('Remove fine scan roughness','SMOOTH'); s.factor=0.5; s.iterations=24; apply(o,s)
# Extra blend around the existing mating rim, using a fitted curved divider.
d=np.load(root/'work/seam_fit.npz'); v=arr(o.data); yz=(v[:,1:]-[0,135])/100
fx=sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(d['coeff'],d['powers']))
w=np.exp(-((v[:,0]-fx)/4.5)**4)
g=o.vertex_groups.new(name='Mating rim blend')
for k in range(1,33):
 ids=np.where((w>(k-1)/32)&(w<=k/32))[0]
 if len(ids): g.add(ids.tolist(),k/32,'REPLACE')
s=o.modifiers.new('Blend irregular original joint','SMOOTH'); s.factor=0.5; s.iterations=70; s.vertex_group=g.name; apply(o,s)
r=o.modifiers.new('Remove collapsed scan fragments','REMESH'); r.mode='VOXEL'; r.voxel_size=0.32; r.adaptivity=0; r.use_smooth_shade=True; apply(o,r)
s=o.modifiers.new('Polish remeshed surface','SMOOTH'); s.factor=.5; s.iterations=8; apply(o,s)
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/union.blend'))
log('Saved union')
m=o.data; m.calc_loop_triangles(); v=arr(m); f=np.empty(len(m.loop_triangles)*3,dtype=np.int32); m.loop_triangles.foreach_get('vertices',f)
np.savez(root/'work/union.npz',v=v,f=f.reshape(-1,3))
log('Exported analysis mesh')
