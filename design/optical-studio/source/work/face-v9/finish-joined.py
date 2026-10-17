import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def load(p):
 bpy.ops.wm.stl_import(filepath=str(p));return bpy.context.object
face=load(out/'face-arms-joined.stl')
# Union the unchanged thickened screen to retain all original optical target vertices.
o=load(r/'face-v8/screen-after.stl');bpy.context.view_layer.objects.active=face;m=face.modifiers.new('Preserve screen','BOOLEAN');m.solver='EXACT';m.use_self=True;m.operation='UNION';m.object=o;bpy.ops.object.modifier_apply(modifier=m.name);assert not face.modifiers;bpy.data.objects.remove(o,do_unlink=True)
print('screen restored',flush=True);rep=export(face,'face-screen-restored');print(rep,flush=True)
