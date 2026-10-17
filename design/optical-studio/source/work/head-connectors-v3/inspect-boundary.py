import bpy,bmesh
from pathlib import Path
p=Path.cwd()/'outputs/danya-simulator/assets/head-connectors-v3/head-neck-clean.stl';bpy.ops.wm.stl_import(filepath=str(p));b=bmesh.new();b.from_mesh(bpy.context.object.data);bmesh.ops.remove_doubles(b,verts=list(b.verts),dist=.0001);bmesh.ops.dissolve_degenerate(b,dist=.0001,edges=list(b.edges));unseen=set(b.verts);cs=[]
while unseen:
 stack=[unseen.pop()];vs=set(stack)
 while stack:
  v=stack.pop()
  for e in v.link_edges:
   w=e.other_vert(v)
   if w in unseen:unseen.remove(w);vs.add(w);stack.append(w)
 cs.append(vs)
for vs in cs:print('COMPONENT',len(vs),[(min(v.co[i]for v in vs),max(v.co[i]for v in vs))for i in range(3)])
e=[e for e in b.edges if not e.is_manifold];print('BAD',len(e));print('BOUNDS',[(min(v.co[i]for x in e for v in x.verts),max(v.co[i]for x in e for v in x.verts))for i in range(3)]if e else None)
