import bpy,bmesh
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/face-v9';bpy.ops.wm.stl_import(filepath=str(r/'face-local-patched.stl'));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5);u=set(bm.verts)
while u:
 st=[u.pop()];vs=set(st)
 while st:
  for e in st.pop().link_edges:
   for v in e.verts:
    if v in u:u.remove(v);vs.add(v);st.append(v)
 print('COMP',len(vs),[min(v.co[i]for v in vs)for i in range(3)],[max(v.co[i]for v in vs)for i in range(3)],flush=True)
