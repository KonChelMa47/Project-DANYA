import numpy as np,bmesh,json
from pathlib import Path
import bpy
p=Path('outputs/danya-simulator/assets/torso-flat-study')
for name in ['head','face']:
 v=np.fromfile(p/(name+'.positions.bin'),np.float32).reshape(-1,3);f=np.fromfile(p/(name+'.indices.bin'),np.uint32).reshape(-1,3)
 m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m)
 print(name,len(v),len(f),'volume',bm.calc_volume(signed=True),'boundary',sum(e.is_boundary for e in bm.edges),'nonmanifold',sum(not e.is_manifold for e in bm.edges),flush=True)
 bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001)
 print('merged',len(bm.verts),len(bm.faces),'volume',bm.calc_volume(signed=True),'boundary',sum(e.is_boundary for e in bm.edges),'nonmanifold',sum(not e.is_manifold for e in bm.edges),flush=True)
 bm.free()
