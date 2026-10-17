import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/thick-head-v3';bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.stl_import(filepath=str(r/'thick-head-clean.stl'));bm=bmesh.new();bm.from_mesh(bpy.context.object.data);unseen=set(bm.verts);rows=[];groups=[]
while unseen:
 stack=[unseen.pop()];vs=set(stack)
 while stack:
  for e in stack.pop().link_edges:
   for v in e.verts:
    if v in unseen:unseen.remove(v);vs.add(v);stack.append(v)
 groups.append(vs)
 rows.append({'vertices':len(vs),'bounds':[[min(v.co[k]for v in vs),max(v.co[k]for v in vs)]for k in range(3)]})
print('COMPONENTS',rows)

