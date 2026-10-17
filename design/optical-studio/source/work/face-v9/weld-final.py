import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/face-v9';bpy.ops.wm.stl_import(filepath=str(r/'face-final.stl'));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data)
for dist in [.0001,.0005,.001]:
 bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=dist);bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=dist);print(dist,sum(not e.is_manifold for e in bm.edges),flush=True)
bm.to_mesh(o.data);bm.free();bpy.ops.wm.stl_export(filepath=str(r/'face-welded.stl'),export_selected_objects=True)
