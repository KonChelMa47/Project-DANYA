import bpy
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'outputs/danya-simulator/assets/tripo-projector';bpy.ops.wm.open_mainfile(filepath=str(p/'projector-calibrated.blend'))
for o in bpy.context.scene.objects:
 if o.type=='MESH':
  for f in o.data.polygons:f.use_smooth=True
bpy.ops.wm.save_as_mainfile(filepath=str(p/'projector-calibrated.blend'))
for o in bpy.context.scene.objects:o.scale*=.001
bpy.ops.export_scene.gltf(filepath=str(p/'projector-calibrated.glb'),export_format='GLB')
