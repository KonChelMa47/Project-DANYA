import bpy,json,math
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'prototype-v7';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.95,.97,.97);s.view_settings.view_transform='Standard';s.render.resolution_x=1400;s.render.resolution_y=900;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=2000;s.camera=cam
for i,n in enumerate(['head-column-left','head-column-right']):
 bpy.ops.wm.stl_import(filepath=str(v/'print'/(n+'.stl')));o=bpy.context.object;o.location.x=(-35 if i==0 else 35);o.color=(.25,.55,.6,1)
bpy.ops.mesh.primitive_plane_add(size=240,location=(0,0,-.1));bpy.context.object.color=(.88,.91,.92,1)
cam.location=(220,-270,250);cam.rotation_euler=(Vector((0,0,0))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=285;s.render.filepath=str(v/'columns-print.png');bpy.ops.render.render(write_still=True)
for o in list(bpy.data.objects):
 if o.type=='MESH':bpy.data.objects.remove(o,do_unlink=True)
bpy.ops.wm.stl_import(filepath=str(v/'print'/'head-column-left.stl'));o=bpy.context.object;o.color=(.25,.55,.6,1)
cam.location=(90,170,70);cam.rotation_euler=(Vector((16,89,10))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=67;s.render.filepath=str(v/'column-hole-detail.png');bpy.ops.render.render(write_still=True)
