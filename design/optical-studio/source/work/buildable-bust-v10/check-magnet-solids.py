import bpy,bmesh,json,numpy as np
from pathlib import Path
root=Path('outputs/danya-simulator/assets/buildable-bust-v10');rows=[]
for p in sorted(root.glob('magnet-embedded-*.positions.bin')):
 if 'tool' in p.name:continue
 v=np.fromfile(p,np.float32).reshape(-1,3);f=np.fromfile(str(p).replace('.positions.bin','.indices.bin'),np.uint32).reshape(-1,3);m=bpy.data.meshes.new(p.stem);m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m);seen=set();sizes=[]
 for vert in bm.verts:
  if vert in seen:continue
  q=[vert];seen.add(vert);n=0
  while q:
   x=q.pop();n+=1
   for e in x.link_edges:
    y=e.other_vert(x)
    if y not in seen:seen.add(y);q.append(y)
  sizes.append(n)
 rows.append({'name':p.name.removesuffix('.positions.bin'),'components':len(sizes),'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'volumeMM3':bm.calc_volume(signed=False)});bm.free();bpy.data.meshes.remove(m)
(root/'embedded-magnet-solids.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
