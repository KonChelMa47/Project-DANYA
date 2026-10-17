import bpy,bmesh,numpy as np,json
from pathlib import Path
from mathutils import Matrix,Vector
p=Path(__file__).resolve().parent;s=json.loads((p/'design.json').read_text())['state'];v=np.fromfile(p/'cad-mesh/engine.local.positions.bin','<f4').reshape(-1,3);bm=bmesh.new()
for co in v:bm.verts.new(co)
bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-4);ret=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False);unused=[x for x in ret['geom_interior']+ret['geom_unused'] if isinstance(x,bmesh.types.BMVert) and x.is_valid];bmesh.ops.delete(bm,geom=list(set(unused)),context='VERTS');bm.normal_update()
for vertex in bm.verts:vertex.co+=vertex.normal*.8
bmesh.ops.triangulate(bm,faces=list(bm.faces));m=bpy.data.meshes.new('Pocket convex envelope');bm.to_mesh(m);bm.free();a=[x*np.pi/180 for x in s['projector']['rotation']];r=Matrix.Rotation(a[0],3,'X')@Matrix.Rotation(a[1],3,'Y')@Matrix.Rotation(a[2],3,'Z')
v=np.array([list(r@x.co+Vector(s['projector']['position'])) for x in m.vertices],dtype='<f4');f=np.array([list(x.vertices) for x in m.polygons],dtype='<u4');v.tofile(p/'pocket.positions.bin');f.tofile(p/'pocket.indices.bin');print(len(v),len(f),flush=True)
