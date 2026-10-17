import bpy,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'))
for p in json.load(open(r/'parts.json'))['parts']:
 if not p['name'].startswith('static-hardware-'):continue
 bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(p['name']+'.stl')));o=bpy.context.object;o.name=p['name'];mat=bpy.data.materials.get('Fasteners')
 if not mat:mat=bpy.data.materials.new('Fasteners');mat.diffuse_color=(.5,.56,.61,1)
 o.data.materials.append(mat)
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
