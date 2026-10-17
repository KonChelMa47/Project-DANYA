import bpy,bmesh,numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent
check=json.loads((p/'cutters-exterior.json').read_text())
assert all(v['atLeast1MM'] for v in check.values()),'Refuse: cutter violates 1 mm head exterior margin'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def load(name,vpath,fpath):
 v=np.fromfile(vpath,'<f4').reshape(-1,3);f=np.fromfile(fpath,'<u4').reshape(-1,3);m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(m);bm.free();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);return o
root=p.parents[2];o=load('head',root/'work/adult-kit/original-head.positions.bin',root/'outputs/danya-simulator/assets/head.indices.bin')
for name in check:
 c=load(name,p/f'{name}.positions.bin',p/f'{name}.indices.bin');bpy.context.view_layer.objects.active=o;mod=o.modifiers.new('Interior '+name,'BOOLEAN');mod.operation='DIFFERENCE';mod.solver='EXACT';mod.object=c;bpy.ops.object.modifier_apply(modifier=mod.name);print(name,len(o.data.vertices),flush=True);assert len(o.data.vertices)>100000,'Unexpected volume removal';bpy.data.objects.remove(c,do_unlink=True)
o.data.calc_loop_triangles();v=np.empty(len(o.data.vertices)*3,np.float32);o.data.vertices.foreach_get('co',v);f=np.empty(len(o.data.loop_triangles)*3,np.int32);o.data.loop_triangles.foreach_get('vertices',f);v.astype('<f4').tofile(p/'head.positions.bin');f.astype('<u4').tofile(p/'head.indices.bin');bpy.ops.wm.save_as_mainfile(filepath=str(p/'internal-cavity-candidate.blend'))
