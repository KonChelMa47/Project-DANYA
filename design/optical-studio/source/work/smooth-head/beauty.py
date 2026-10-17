import bpy
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=Path.cwd()/'work/smooth-head';bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;objs={}
for n in ['head-upper','head-lower','magnet-carrier','face']:
 bpy.ops.wm.stl_import(filepath=str((r/'assembly'if n=='face'else r/'solid-seam-v2'if n=='magnet-carrier'else r/'smooth-head-v4')/(n+'.stl')));o=bpy.context.object;o.name=n;o.color=(.8,.8,.76,1);objs[n]=o
s.world=bpy.data.worlds.new('World');s.world.color=(.8,.8,.8);s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.render.resolution_x=1400;s.render.resolution_y=1400;s.render.resolution_percentage=100;s.view_settings.view_transform='Standard';bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=3000;s.camera=cam
for name,vis,loc,target,scale in [('assembled',['head-upper','head-lower','magnet-carrier','face'],(360,-600,180),(0,0,165),260),('assembled-side',['head-upper','head-lower','magnet-carrier','face'],(0,-600,170),(0,0,165),260)]:
 for n,o in objs.items():o.hide_render=n not in vis
 cam.location=loc;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;s.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
