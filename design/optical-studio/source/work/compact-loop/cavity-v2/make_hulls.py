import bpy,bmesh,json,numpy as np,sys
from pathlib import Path
from itertools import product
p=Path(__file__).resolve().parent

def hull(coords):
 bm=bmesh.new()
 for co in set(tuple(round(float(x),5) for x in row) for row in coords):bm.verts.new(co)
 ret=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False)
 unused=[x for x in ret['geom_interior']+ret['geom_unused'] if isinstance(x,bmesh.types.BMVert) and x.is_valid]
 bmesh.ops.delete(bm,geom=list(set(unused)),context='VERTS')
 bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
 return bm

for path in sorted(p.glob('wall_*.points.json' if '--walls-only' in sys.argv else '*.points.json')):
 name=path.name.removesuffix('.points.json');data=json.loads(path.read_text());bm=hull(data['points']);base=[tuple(v.co) for v in bm.verts];bm.free();pad=data['pad']
 # Minkowski sum with a cube: conservative clearance in every direction.
 bm=hull([[co[i]+shift[i] for i in range(3)] for co in base for shift in product([-pad,pad],repeat=3)])
 bmesh.ops.triangulate(bm,faces=list(bm.faces));m=bpy.data.meshes.new(name);bm.to_mesh(m);bm.free()
 np.array([list(v.co) for v in m.vertices],dtype='<f4').tofile(p/f'{name}.positions.bin')
 np.array([list(f.vertices) for f in m.polygons],dtype='<u4').tofile(p/f'{name}.indices.bin')
 print(name,len(m.vertices),len(m.polygons),flush=True)
