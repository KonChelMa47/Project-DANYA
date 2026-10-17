import bpy,numpy as np,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
bpy.ops.wm.open_mainfile(filepath=str(root/'work/interior_candidate.blend'))
o=bpy.data.objects['DANYA-face-back'];m=o.data
def coords(m,field='co'):
 a=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get(field,a);return a.reshape(-1,3)
v=coords(m);n=coords(m,'normal');rad=v-[-20,0,140];rad/=np.maximum(np.linalg.norm(rad,axis=1)[:,None],1e-10)
inside=-np.sum(rad*n,axis=1);w=np.clip((inside-.05)/.4,0,1)
fit=np.load(root/'work/seam_fit.npz');yz=(v[:,1:]-[0,135])/100;fx=sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
w*=np.clip((abs(v[:,0]-fx)-4)/4,0,1);w*=np.clip((v[:,2]-43)/5,0,1)
oldnorm=np.empty(len(m.loops)*3,dtype=np.float32);m.corner_normals.foreach_get('vector',oldnorm);oldnorm=oldnorm.reshape(-1,3)
g=o.vertex_groups.new(name='Interior finish only')
for k in range(1,33):
 ids=np.where((w>(k-1)/32)&(w<=k/32))[0]
 if len(ids):g.add(ids.tolist(),k/32,'REPLACE')
bpy.context.view_layer.objects.active=o
mod=o.modifiers.new('Polish inner walls and remove small bumps','SMOOTH');mod.factor=.5;mod.iterations=100;mod.vertex_group=g.name
bpy.ops.object.modifier_apply(modifier=mod.name);m=o.data;after=coords(m);delta=np.linalg.norm(after-v,axis=1)
assert np.all(delta[w==0]==0),'Protected exterior or interface moved'
print('Displacement',np.percentile(delta[w>0],[50,95,99,100]).tolist(),flush=True)
m.normals_split_custom_set(np.zeros((len(m.loops),3)).tolist());m.update()
norms=np.empty(len(m.loops)*3,dtype=np.float32);m.corner_normals.foreach_get('vector',norms);norms=norms.reshape(-1,3)
for p in m.polygons:
 if all(delta[i]==0 for i in p.vertices):norms[p.loop_start:p.loop_start+p.loop_total]=oldnorm[p.loop_start:p.loop_start+p.loop_total]
m.normals_split_custom_set(norms.tolist());m.update();m.calc_loop_triangles()
f=np.empty(len(m.loop_triangles)*3,dtype=np.int32);m.loop_triangles.foreach_get('vertices',f)
np.savez(root/'work/interior_back.npz',v=after,f=f.reshape(-1,3))
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/interior_candidate.blend'))
(root/'work/interior_polish.json').write_text(json.dumps({'protected_vertex_count':int(sum(w==0)),'protected_vertex_max_movement':float(delta[w==0].max()),'interior_movement_percentiles':np.percentile(delta[w>0],[50,95,99,100]).tolist()},indent=2))
