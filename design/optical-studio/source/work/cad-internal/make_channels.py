import bpy,bmesh,json,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
for i,points in enumerate(json.loads((p/'channel-points.json').read_text())):
 bm=bmesh.new()
 for co in set(tuple(round(x,4) for x in row) for row in points):bm.verts.new(co)
 ret=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False);unused=[x for x in ret['geom_interior']+ret['geom_unused'] if isinstance(x,bmesh.types.BMVert) and x.is_valid];bmesh.ops.delete(bm,geom=list(set(unused)),context='VERTS');bm.normal_update()
 if i<3:
  for v in bm.verts:v.co+=v.normal*.4
 bmesh.ops.triangulate(bm,faces=list(bm.faces));m=bpy.data.meshes.new('channel');bm.to_mesh(m);bm.free();np.array([list(v.co) for v in m.vertices],dtype='<f4').tofile(p/f'channel{i}.positions.bin');np.array([list(f.vertices) for f in m.polygons],dtype='<u4').tofile(p/f'channel{i}.indices.bin');print(i,len(m.vertices),flush=True)
