import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(out/'face-local-patched.stl'));face=bpy.context.object
for i in range(4):
 bpy.ops.wm.stl_import(filepath=str(out/f'face-holder-{i}.stl'));o=bpy.context.object;bpy.context.view_layer.objects.active=face;m=face.modifiers.new('Union real arm','BOOLEAN');m.solver='EXACT';m.use_self=True;m.operation='UNION';m.object=o;bpy.ops.object.modifier_apply(modifier=m.name);assert len(face.modifiers)==0;bpy.data.objects.remove(o,do_unlink=True);print('joined arm',i,flush=True)
rep=export(face,'face-arms-joined');print(rep,flush=True)
