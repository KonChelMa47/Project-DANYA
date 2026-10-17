import bpy,bmesh
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/face-v9';bpy.ops.wm.stl_import(filepath=str(r/'face-final.stl'));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5)
for e in bm.edges:
 if not e.is_manifold:print('BAD',len(e.link_faces),[v.co[:]for v in e.verts], [(f.calc_area(), len(f.verts))for f in e.link_faces],flush=True)
