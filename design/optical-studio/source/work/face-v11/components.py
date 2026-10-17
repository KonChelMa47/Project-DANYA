import bpy,bmesh
from pathlib import Path
for name in ['magnet-carrier-revised','head-lower-release']:
 bpy.ops.wm.stl_import(filepath=str(Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/face-v11'/(name+'.stl')));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);left=set(bm.verts);cs=[]
 while left:
  comp={left.pop()};stack=list(comp)
  while stack:
   v=stack.pop()
   for e in v.link_edges:
    w=e.other_vert(v)
    if w in left:left.remove(w);comp.add(w);stack.append(w)
  cs.append((len(comp),[min(v.co[k]for v in comp)for k in range(3)],[max(v.co[k]for v in comp)for k in range(3)]))
 print(name,sorted(cs,reverse=True),flush=True)
