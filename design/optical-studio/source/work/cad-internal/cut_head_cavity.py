import bpy,bmesh,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
objects=[]
for name in ['head','cavity']:
 v=np.fromfile(p/f'{name}.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/f'{name}.indices.bin','<u4').reshape(-1,3)
 m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(m);bm.free();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);objects.append(o)
o,c=objects;bpy.context.view_layer.objects.active=o;mod=o.modifiers.new('Continuous optical cavity','BOOLEAN');mod.operation='DIFFERENCE';mod.solver='EXACT';mod.object=c;print('Boolean start',flush=True);bpy.ops.object.modifier_apply(modifier=mod.name);print('Boolean done',flush=True)
o.data.calc_loop_triangles();v=np.empty(len(o.data.vertices)*3,np.float32);o.data.vertices.foreach_get('co',v);f=np.empty(len(o.data.loop_triangles)*3,np.int32);o.data.loop_triangles.foreach_get('vertices',f);v.reshape(-1,3).astype('<f4').tofile(p/'head.positions.bin');f.reshape(-1,3).astype('<u4').tofile(p/'head.indices.bin');print(len(v)//3,len(f)//3,flush=True)
