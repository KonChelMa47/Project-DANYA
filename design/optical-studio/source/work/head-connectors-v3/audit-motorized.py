import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/head-connectors-v3';rows=[]
for p in (r/'motorized-print').glob('*.stl'):
 bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;b=bmesh.new();b.from_mesh(o.data);bmesh.ops.remove_doubles(b,verts=list(b.verts),dist=.00001);bad=sum(not e.is_manifold for e in b.edges);lo=[min(v.co[i]for v in b.verts)for i in range(3)];hi=[max(v.co[i]for v in b.verts)for i in range(3)];rows.append({'file':p.name,'nonManifoldEdges':bad,'dimensionsMM':[hi[i]-lo[i]for i in range(3)],'bottomZ':lo[2]});assert bad==0 and abs(lo[2])<.001;b.free()
(r/'motorized-print-audit.json').write_text(json.dumps(rows,indent=2));print(rows)
