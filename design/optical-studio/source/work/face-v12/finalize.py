exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
for name in ['magnet-carrier-covered','head-upper-covered','head-lower-covered']:
 o=load(out/(name+'.stl'));clean(o);bm=bmesh.new();bm.from_mesh(o.data);left=set(bm.verts);cs=[]
 while left:
  comp={left.pop()};stack=list(comp)
  while stack:
   v=stack.pop()
   for e in v.link_edges:
    w=e.other_vert(v)
    if w in left:left.remove(w);comp.add(w);stack.append(w)
  cs.append(comp)
 cs.sort(key=len,reverse=True)
 for c in cs[1:]:bmesh.ops.delete(bm,geom=list(c),context='VERTS')
 for _ in range(20):
  bad=[e for e in bm.edges if not e.is_manifold]
  remove={f for e in bad if e.is_boundary for f in e.link_faces if f.calc_area()<.01 and any(len(ee.link_faces)>2 for ee in f.edges)}
  if not remove:break
  bmesh.ops.delete(bm,geom=list(remove),context='FACES_ONLY');wire=[e for e in bm.edges if not e.link_faces]
  if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
 bm.to_mesh(o.data);bm.free();print(name,export(o,name),flush=True)
