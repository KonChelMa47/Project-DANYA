import bpy
from pathlib import Path
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True);r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/ad5m-one-plate';s=bpy.context.scene
for name,col in [('head-upper',(.25,.61,.69,1)),('head-lower',(.86,.59,.3,1))]:
 bpy.ops.wm.stl_import(filepath=str(r/(name+'-AD5M.stl')));bpy.context.object.color=col
bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,-2));bed=bpy.context.object;bed.dimensions=(220,220,4);bed.color=(.32,.35,.4,1)
s.world=bpy.data.worlds.new('World');s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.94,.95,.97);s.render.resolution_x=1100;s.render.resolution_y=1000;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.view_settings.view_transform='Standard';s.view_settings.look='None'
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=3000;s.camera=cam
for name,loc,scale,target in [('plate-view',(340,-450,390),360,(0,0,40)),('plate-top',(0,0,600),255,(0,0,0))]:
 cam.location=loc;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;s.render.filepath=str(r/(name+'.png'));bpy.ops.render.render(write_still=True)
