import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
def load(p):
 bpy.ops.wm.stl_import(filepath=str(p));return bpy.context.object
def boolean(a,b,op,remove=True):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);assert not a.modifiers
 if remove:bpy.data.objects.remove(b,do_unlink=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
collar=load(out/'collar-clear.stl');bm=bmesh.new();bm.from_mesh(collar.data);u=set(bm.verts);groups=[]
while u:
 st=[u.pop()];vs=set(st)
 while st:
  for e in st.pop().link_edges:
   for v in e.verts:
    if v in u:u.remove(v);vs.add(v);st.append(v)
 groups.append(vs)
groups.sort(key=len,reverse=True)
for vs in groups[1:]:
 ext=[max(v.co[i]for v in vs)-min(v.co[i]for v in vs)for i in range(3)];print('component',len(vs),ext,flush=True)
 if len(vs)<500 and max(ext)<4:bmesh.ops.delete(bm,geom=list(vs),context='VERTS')
bm.to_mesh(collar.data);bm.free();print('clean collar',export(collar,'collar-clear-clean'),flush=True)
face=load(r/'face-v8/screen-after.stl');boolean(face,collar,'UNION');rep=export(face,'face-preserved-screen');(out/'preserved-screen-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
