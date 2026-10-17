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
collar=load(out/'face-final.stl');cut=load(out/'collar-partition-tool.stl');cut.location.x=6.6;boolean(collar,cut,'DIFFERENCE');print('isolated collar',export(collar,'collar-only'),flush=True)
beam=load(out/'beam-volume.stl')
for offset in [(0,0,0),(.15,0,0),(-.15,0,0),(0,.15,0),(0,-.15,0),(0,0,.15),(0,0,-.15)]:
 beam.location=offset;boolean(collar,beam,'DIFFERENCE',False)
print('carved collar',export(collar,'collar-clear'),flush=True)
face=load(r/'face-v8/screen-after.stl');boolean(face,collar,'UNION');rep=export(face,'face-preserved-screen');(out/'preserved-screen-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
