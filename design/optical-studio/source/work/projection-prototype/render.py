import bpy,math
from mathutils import Vector
from pathlib import Path
root=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';bpy.ops.wm.open_mainfile(filepath=str(root/'projection-prototype.blend'))
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.studiolight_rotate_z=.5;s.display.shading.color_type='MATERIAL';s.display.shading.show_shadows=True;s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.background_type='WORLD';s.world.color=(.065,.085,.1);s.render.resolution_x=1100;s.render.resolution_y=1100;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.film_transparent=False
bpy.ops.object.camera_add(location=(700,-900,420));cam=bpy.context.object;cam.rotation_euler=(Vector((0,0,45))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=560;cam.data.clip_end=3000;s.camera=cam
for screen in bpy.data.screens:
 for area in screen.areas:
  if area.type=='VIEW_3D':
   area.spaces.active.region_3d.view_location=(0,0,45);area.spaces.active.region_3d.view_distance=650;area.spaces.active.region_3d.view_rotation=cam.rotation_euler.to_quaternion()
bpy.ops.wm.save_as_mainfile(filepath=str(root/'projection-prototype.blend'))
for name in ['overview','open-head']:
 bpy.data.objects['head'].hide_render=True;
 for part in ['head-upper','head-lower']:bpy.data.objects[part].hide_render=name=='open-head'
 bpy.data.objects['face'].hide_render=name=='open-head';s.render.filepath=str(root/(name+'.png'));bpy.ops.render.render(write_still=True)
