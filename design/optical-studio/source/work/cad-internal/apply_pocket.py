import bpy,bmesh,numpy as np,shutil,json
from pathlib import Path
p=Path(__file__).resolve().parent
check=json.loads((p/'pocket-exterior.json').read_text());assert not any(check.values()),'Pocket touches exterior'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);objects=[]
for name in ['head','pocket']:
 v=np.fromfile(p/f'{name}.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/f'{name}.indices.bin','<u4').reshape(-1,3);m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);objects.append(o)
for suffix in ['positions','indices']:shutil.copy2(p/f'head.{suffix}.bin',p/f'head-before-pocket.{suffix}.bin')
o,c=objects;bpy.context.view_layer.objects.active=o;mod=o.modifiers.new('CAD engine convex clearance pocket','BOOLEAN');mod.operation='DIFFERENCE';mod.solver='EXACT';mod.object=c;print('Pocket boolean start',flush=True);bpy.ops.object.modifier_apply(modifier=mod.name)
bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-7);bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-7);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(o.data);bm.free();o.data.calc_loop_triangles();v=np.empty(len(o.data.vertices)*3,np.float32);o.data.vertices.foreach_get('co',v);f=np.empty(len(o.data.loop_triangles)*3,np.int32);o.data.loop_triangles.foreach_get('vertices',f);v.astype('<f4').tofile(p/'head.positions.bin');f.astype('<u4').tofile(p/'head.indices.bin');print(len(v)//3,len(f)//3,flush=True)
