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
collar=load(out/'collar-clear-clean.stl');bm=bmesh.new();bm.from_mesh(collar.data)
for iteration in range(20):
 bad=[e for e in bm.edges if not e.is_manifold];print('bad edges',len(bad),flush=True)
 if not bad:break
 remove={f for e in bad if e.is_boundary for f in e.link_faces if f.calc_area()<.00001}
 if not remove:break
 bmesh.ops.delete(bm,geom=list(remove),context='FACES_ONLY');wire=[e for e in bm.edges if not e.link_faces]
 if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
bm.to_mesh(collar.data);bm.free();rep=export(collar,'collar-ready');print(rep,flush=True);assert rep['nonManifoldEdges']==0
face=load(r/'face-v8/screen-after.stl');boolean(face,collar,'UNION');rep=export(face,'face-preserved-screen');(out/'preserved-screen-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
