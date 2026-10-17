import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/head-split-quarter';bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.stl_import(filepath=str(r/'head-upper.stl'));bm=bmesh.new();bm.from_mesh(bpy.context.object.data);unseen=set(bm.verts);rows=[];groups=[]
while unseen:
 stack=[unseen.pop()];vs=set(stack)
 while stack:
  for e in stack.pop().link_edges:
   for v in e.verts:
    if v in unseen:unseen.remove(v);vs.add(v);stack.append(v)
 groups.append(vs)
 rows.append({'vertices':len(vs),'bounds':[[min(v.co[k]for v in vs),max(v.co[k]for v in vs)]for k in range(3)]})
print('COMPONENTS',rows)

from mathutils.bvhtree import BVHTree
large=max(groups,key=len);small=min(groups,key=len);verts=list(large);lookup={v:i for i,v in enumerate(verts)};faces={f for v in verts for f in v.link_faces};tree=BVHTree.FromPolygons([v.co for v in verts],[[lookup[v]for v in f.verts]for f in faces]);pairs=[]
for v in small:
 p,n,i,d=tree.find_nearest(v.co);pairs.append({'a':list(v.co),'b':list(p),'distance':d})
pairs.sort(key=lambda p:p['distance']);(r/'component-gap.json').write_text(json.dumps(pairs,indent=2));print('NEAREST',pairs[:8])
