import bpy
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'printable-seam';bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;objs={}
for n in ['head-upper','head-lower','magnet-carrier','face']:
 bpy.ops.wm.stl_import(filepath=str((r/'assembly'if n=='face'else out)/(n+'.stl')));o=bpy.context.object;o.name=n;o.color=(.89,.86,.78,1)if n=='face'else(.28,.61,.67,1)if n=='magnet-carrier'else(.68,.72,.72,1);objs[n]=o
s.world=bpy.data.worlds.new('World');s.world.color=(.94,.96,.97);s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.render.resolution_x=1200;s.render.resolution_y=1000;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.view_settings.view_transform='Standard';s.view_settings.look='None'
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=3000;s.camera=cam
for name,loc,target,scale,off,face in [('assembled',(430,-600,245),(0,0,165),290,0,True),('exploded',(430,-600,245),(15,0,165),330,30,True),('seam-detail',(155,-440,180),(5,-65,165),80,6,False)]:
 for n in ['face','magnet-carrier']:objs[n].location.x=off
 objs['face'].hide_render=not face;cam.location=loc;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;s.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
