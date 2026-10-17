import bpy,numpy as np,bmesh
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['baseline','cavity']:
 v=np.fromfile(p/f'{name}.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/f'{name}.indices.bin','<u4').reshape(-1,3)
 m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m)
 print(name,'nonmanifold',sum(not e.is_manifold for e in bm.edges),'inconsistent',sum(e.is_manifold and not e.is_contiguous for e in bm.edges),'volume',bm.calc_volume(signed=True),flush=True)
 bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));print('recalced volume',bm.calc_volume(signed=True),flush=True);bm.free()
