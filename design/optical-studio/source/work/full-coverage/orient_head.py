import bpy,bmesh,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;v=np.fromfile(p/'head.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/'joint-head.indices.bin','<u4').reshape(-1,3);m=bpy.data.meshes.new('head');m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));print('volume',bm.calc_volume(signed=True),'nonmanifold',sum(not e.is_manifold for e in bm.edges),flush=True);bm.to_mesh(m);bm.free();m.calc_loop_triangles();f=np.empty(len(m.loop_triangles)*3,np.uint32);m.loop_triangles.foreach_get('vertices',f);backup=p/'head-before-retriangulation.indices.bin';
if not backup.exists():backup.write_bytes((p/'head.indices.bin').read_bytes())
f.tofile(p/'head.indices.bin')
