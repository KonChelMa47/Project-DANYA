import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';bpy.ops.wm.read_factory_settings(use_empty=True);result={}
parts=[p for p in json.loads((r/'parts.json').read_text())['parts']if p.get('print')]
for p in parts:
 n=p['name'];folder=r/'printable-seam'if n in ['head-upper','head-lower','magnet-carrier']else r/'head-split-quarter'if n.startswith('head-alignment-pin-')else r/'assembly';path=folder/(n+'.stl');bpy.ops.wm.stl_import(filepath=str(path));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);unseen=set(bm.verts);groups=[]
 while unseen:
  vs={unseen.pop()};stack=list(vs)
  while stack:
   for e in stack.pop().link_edges:
    for v in e.verts:
     if v in unseen:unseen.remove(v);vs.add(v);stack.append(v)
  faces=set(f for v in vs for f in v.link_faces);groups.append({'vertices':len(vs),'faces':len(faces),'area':sum(f.calc_area()for f in faces),'bounds':[[min(v.co[k]for v in vs),max(v.co[k]for v in vs)]for k in range(3)]})
 groups.sort(key=lambda p:p['vertices'],reverse=True);result[n]={'source':str(path),'components':groups,'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges)};print(n,len(groups),[g['vertices']for g in groups],flush=True);bm.free();bpy.data.objects.remove(o,do_unlink=True)
Path('work/assembly-integrity/components.json').write_text(json.dumps(result,indent=2))
