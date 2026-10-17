import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/'collar-exact-clear.stl'));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);u=set(bm.verts);groups=[]
while u:
 st=[u.pop()];vs=set(st)
 while st:
  for e in st.pop().link_edges:
   for v in e.verts:
    if v in u:u.remove(v);vs.add(v);st.append(v)
 groups.append(vs)
groups.sort(key=len,reverse=True)
for k,vs in enumerate(groups):
 lo=[min(v.co[i]for v in vs)for i in range(3)];hi=[max(v.co[i]for v in vs)for i in range(3)];print('component',k,len(vs),lo,hi,flush=True)
 if k:bmesh.ops.delete(bm,geom=list(vs),context='VERTS')
bm.to_mesh(o.data);bm.free();print(export(o,'collar-exact-main'),flush=True)
bpy.ops.wm.stl_import(filepath=str(r/'face-v8/screen-after.stl'));face=bpy.context.object;bpy.context.view_layer.objects.active=face;m=face.modifiers.new('Exact collar junction','BOOLEAN');m.operation='UNION';m.solver='EXACT';m.object=o;bpy.ops.object.modifier_apply(modifier=m.name);assert not face.modifiers;rep=export(face,'face-exact-union');(out/'exact-union-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
