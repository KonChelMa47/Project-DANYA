import bpy
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype'
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'))
for n in ['foldMirror-glass','foldMirror-cutting-master']:
 old=bpy.data.objects.get(n)
 assert old is not None,n
 mat=old.data.materials[0] if old.data.materials else None
 bpy.data.objects.remove(old,do_unlink=True)
 bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(n+'.stl')));o=bpy.context.object;o.name=n
 if mat:o.data.materials.append(mat)
 if n.endswith('master'):o.hide_render=True;o.hide_set(True)
 o['revision']='small-mirror-v1';o['scope']='Same reflector plane and mount datum; size and cutting template only.'
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
