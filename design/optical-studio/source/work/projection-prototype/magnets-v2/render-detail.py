import bpy,math
from pathlib import Path
from mathutils import Vector,Matrix
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'));s=bpy.context.scene
for o in s.objects:o.hide_render=True
for side,dy in [('head',-100),('face',100)]:
 M=Matrix.Translation((0,dy,0))
 if side=='face':M=M@Matrix.Translation((40,0,160))@Matrix.Rotation(math.pi,4,'Z')@Matrix.Translation((-40,0,-160))
 for name in [side]+[f'magnet-clear-{i}-{side}-magnet'for i in range(4)]:
  o=bpy.data.objects[name];o.hide_render=False;o.matrix_world=M@o.matrix_world
bpy.ops.object.camera_add(location=(600,0,300));cam=bpy.context.object;cam.rotation_euler=(Vector((0,0,160))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=415;cam.data.clip_end=3000;s.camera=cam
s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.065,.085,.1);s.render.resolution_x=1400;s.render.resolution_y=950;s.render.resolution_percentage=100;s.render.filepath=str(r/'magnet-detail.png');bpy.ops.render.render(write_still=True)
