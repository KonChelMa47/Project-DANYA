import bpy,bmesh,sys,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/unified-head-v5';name=sys.argv[-1];bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.stl_import(filepath=str(r/(name+'.stl')));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data)
bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.001);bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=.0001)
# Close only tiny numerical boundary loops, not large functional openings.
for _ in range(3):
 boundaries=[e for e in bm.edges if e.is_boundary];bmesh.ops.holes_fill(bm,edges=boundaries,sides=8);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.0001)
# Remove isolated numerical fragments smaller than1cubicmm bounding box.
unseen=set(bm.verts);remove=[];groups=[]
while unseen:
 stack=[unseen.pop()];vs=set(stack)
 while stack:
  for e in stack.pop().link_edges:
   for v in e.verts:
    if v in unseen:unseen.remove(v);vs.add(v);stack.append(v)
 lo=[min(v.co[i]for v in vs)for i in range(3)];hi=[max(v.co[i]for v in vs)for i in range(3)];vol=(hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]);groups.append((len(vs),vol))
 if len(vs)<100 and vol<1:remove+=list(vs)
if remove:bmesh.ops.delete(bm,geom=remove,context='VERTS')
bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));print('groups',groups,'nonM',sum(not e.is_manifold for e in bm.edges),flush=True);bm.to_mesh(o.data);bm.free();bpy.ops.wm.stl_export(filepath=str(r/(name+'-weld.stl')),export_selected_objects=True)
