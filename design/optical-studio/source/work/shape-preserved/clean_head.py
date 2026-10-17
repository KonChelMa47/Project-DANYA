import bpy,bmesh,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
for suffix in ['positions','indices']:
 b=p/f'head-before-clean.{suffix}.bin'
 if not b.exists():b.write_bytes((p/f'head.{suffix}.bin').read_bytes())
v=np.fromfile(p/'head-before-clean.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/'head-before-clean.indices.bin','<u4').reshape(-1,3);m=bpy.data.meshes.new('head');m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6);bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-7);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));print('nonmanifold',sum(not e.is_manifold for e in bm.edges),flush=True);bm.to_mesh(m);bm.free();m.calc_loop_triangles();v=np.empty(len(m.vertices)*3,np.float32);m.vertices.foreach_get('co',v);f=np.empty(len(m.loop_triangles)*3,np.uint32);m.loop_triangles.foreach_get('vertices',f);v.tofile(p/'head.positions.bin');f.tofile(p/'head.indices.bin');print('vertices',len(v)//3,'triangles',len(f)//3,flush=True)
