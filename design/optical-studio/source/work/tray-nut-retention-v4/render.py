import bpy,bmesh,json,math
from pathlib import Path
from mathutils import Matrix,Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'tray-nut-retention-v4';data=json.load(open(v/'parts.json'));pm=Matrix([[data['matrix'][i+j*4]for j in range(4)]for i in range(4)]);inv=pm.inverted();bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def load(n,local=False):
 bpy.ops.wm.stl_import(filepath=str(v/(n+'.stl')));o=bpy.context.object;o.name=n
 if not local:
  for p in o.data.vertices:p.co=inv@p.co
 o.color=(.2,.48,.51,1)if n=='drop-projector-tray-t5' or 'coupon'in n else(.96,.59,.2,1)if'cover'in n else(.65,.7,.74,1)
 return o
tray=load('drop-projector-tray-t5');caps=[load('tray-nut-cover-'+str(i))for i in range(4)];nuts=[load('drop-projector-tray-t5-frame-nut-'+str(i))for i in range(4)]
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.93,.95,.95);s.view_settings.view_transform='Standard';s.render.resolution_x=1300;s.render.resolution_y=900;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=2000;s.camera=cam
cam.location=(160,-210,150);cam.rotation_euler=(Vector((67.5,-5,37.5))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=190;s.render.filepath=str(v/'tray-underside.png');bpy.ops.render.render(write_still=True)
# Show a cap outside its seat and a seated nut; spread only for explanatory diagram.
for o in [tray,*caps,*nuts]:o.hide_render=True
seat=load('tray-nut-fit-coupon',True);cap=load('tray-nut-cover-local',True);cap.location=(0,-3,14);nut=nuts[0].copy();nut.data=nuts[0].data.copy();bpy.context.collection.objects.link(nut);nut.hide_render=False
for p in nut.data.vertices:p.co.x-=12;p.co.z-=7
cam.location=(27,-44,31);cam.rotation_euler=(Vector((0,-5,6))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=39;s.render.filepath=str(v/'nut-cover-detail.png');bpy.ops.render.render(write_still=True)
