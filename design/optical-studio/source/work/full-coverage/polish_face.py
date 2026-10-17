import bpy,numpy as np,json,sys
from pathlib import Path
p=Path(__file__).resolve().parent
v=np.fromfile(p/'face.positions.bin','<f4').reshape(-1,3).copy();f=np.fromfile(p/'face.indices.bin','<u4').reshape(-1,3);mask=np.fromfile(p/'face-inner-mask.bin','u1')>0
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
m=bpy.data.meshes.new('face');m.from_pydata(v.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new('face',m);bpy.context.collection.objects.link(o);bpy.context.view_layer.objects.active=o;g=o.vertex_groups.new(name='Interior rim only');g.add(np.where(mask)[0].tolist(),1,'REPLACE')
mod=o.modifiers.new('Remove inner corrugation','SMOOTH');mod.factor=.5;mod.iterations=int(sys.argv[-1]);mod.vertex_group=g.name;bpy.ops.object.modifier_apply(modifier=mod.name)
w=np.empty(v.size,np.float32);o.data.vertices.foreach_get('co',w);w=w.reshape(-1,3);delta=w-v;length=np.linalg.norm(delta,axis=1);w=v+delta*np.minimum(1,2.5/np.maximum(length,1e-9))[:,None]
assert np.max(np.abs(w[~mask]-v[~mask]))==0
w.astype('<f4').tofile(p/'face.positions.bin');np.savez(p/'face-design.npz',v=w,f=f);print('Inner movement',np.percentile(np.linalg.norm(w-v,axis=1)[mask],[50,95,100]),flush=True)
