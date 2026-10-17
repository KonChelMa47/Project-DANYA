import bpy,bmesh,numpy as np,json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/buildable-bust-v10');v=np.fromfile(r/'body-wall3.positions.bin',np.float32).reshape(-1,3);f=np.fromfile(r/'body-wall3.indices.bin',np.uint32).reshape(-1,3);m=bpy.data.meshes.new('body-wall3');m.from_pydata(v.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new('body-wall3',m);bpy.context.collection.objects.link(o);bm=bmesh.new();bm.from_mesh(m);seen=set();components=0
for p in bm.verts:
 if p in seen:continue
 components+=1;q=[p];seen.add(p)
 while q:
  a=q.pop()
  for e in a.link_edges:
   b=e.other_vert(a)
   if b not in seen:seen.add(b);q.append(b)
result={'components':components,'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'signedVolumeMM3':bm.calc_volume(signed=True),'nominalWallMM':3,'scope':'Topology/volume only, not a self-intersection or strength qualification.'};bm.free();bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(r/'body-wall3.stl'),export_selected_objects=True);(r/'body-wall3-mesh-check.json').write_text(json.dumps(result,indent=2));print(result)
