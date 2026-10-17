import bpy,bmesh,json,shutil
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/printable-seam';bpy.ops.wm.read_factory_settings(use_empty=True);report={}
for n in ['head-upper','head-lower','magnet-carrier']:
 bpy.ops.wm.stl_import(filepath=str(r/(n+'-fixed.stl')));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);unseen=set(bm.verts);groups=[]
 while unseen:
  vs={unseen.pop()};stack=list(vs)
  while stack:
   for e in stack.pop().link_edges:
    for v in e.verts:
     if v in unseen:unseen.remove(v);vs.add(v);stack.append(v)
  groups.append(vs)
 groups.sort(key=len,reverse=True);removed=[]
 for g in groups[1:]:
  bounds=[[min(v.co[k]for v in g),max(v.co[k]for v in g)]for k in range(3)];removed.append({'vertices':len(g),'bounds':bounds});assert len(g)<500,removed;bmesh.ops.delete(bm,geom=list(g),context='VERTS')
 bad=sum(not e.is_manifold for e in bm.edges);assert bad==0;bm.to_mesh(o.data);bm.free();bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(r/(n+'.stl')),export_selected_objects=True);report[n]={'components':1,'nonManifoldEdges':bad,'removedSmallFragments':removed};bpy.data.objects.remove(o,do_unlink=True)
(r/'geometry-final.json').write_text(json.dumps(report,indent=2));print(report)
