import bpy,bmesh,json
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/solid-seam-v2';bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.stl_import(filepath=str(r/'head-lower.stl'));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);bm.verts.ensure_lookup_table();orig=[v.co.copy()for v in bm.verts];weights=[]
for p in orig:weights.append(max(0,min(1,(78-p.z)/12))*max(0,min(1,(-p.x-1)/8))*max(0,min(1,(40-abs(p.y))/10)))
for _ in range(40):
 coords=[v.co.copy()for v in bm.verts]
 for v,w in zip(bm.verts,weights):
  if w==0:continue
  ns=[e.other_vert(v).index for e in v.link_edges];mean=sum((coords[j]for j in ns),Vector())/len(ns);p=coords[v.index].lerp(mean,.5*w);delta=p-orig[v.index]
  if delta.length>.6:delta*=.6/delta.length
  p=orig[v.index]+delta
  if orig[v.index].z<50.6501:p.z=50.65
  v.co=p
bm.normal_update();report={'region':'neck vertices belowZ78 and behindX-1, smoothly weighted; bottom plane constrained','iterations':40,'maximumDisplacementMM':max((v.co-p).length for v,p in zip(bm.verts,orig)),'changedVertices':sum((v.co-p).length>1e-6 for v,p in zip(bm.verts,orig))};bm.to_mesh(o.data);bm.free();bpy.ops.wm.stl_export(filepath=str(r/'head-lower.stl'),export_selected_objects=True);(r/'neck-smoothing.json').write_text(json.dumps(report,indent=2));print(report)
