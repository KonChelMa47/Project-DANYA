import bpy,bmesh,json,hashlib
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';rows=[]
for p in sorted((r/'print').glob('*.stl')):
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001)
 unseen=set(bm.verts);components=[]
 while unseen:
  stack=[unseen.pop()];count=0
  while stack:
   v=stack.pop();count+=1
   for e in v.link_edges:
    w=e.other_vert(v)
    if w in unseen:unseen.remove(w);stack.append(w)
  components.append(count)
 lo=[min(v.co[i]for v in bm.verts)for i in range(3)];hi=[max(v.co[i]for v in bm.verts)for i in range(3)]
 rows.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'connectedComponents':len(components),'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'dimensionsMM':[hi[i]-lo[i]for i in range(3)],'bottomZ':lo[2]});bm.free()
(r/'final-print-audit.json').write_text(json.dumps(rows,indent=2));print('ISSUES',[(a['file'],a['connectedComponents'],a['nonManifoldEdges'])for a in rows if a['connectedComponents']!=1 or a['nonManifoldEdges']]);print('TOTAL',len(rows))
