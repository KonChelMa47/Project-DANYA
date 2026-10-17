import bpy,json,math
from pathlib import Path
from mathutils import Matrix,Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'printable-mounts-v5';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.93,.95,.95);s.view_settings.view_transform='Standard';s.render.resolution_x=1300;s.render.resolution_y=1000;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=2000;s.camera=cam
pm=json.load(open(v/'tray-parts.json'))['matrix'];pm=Matrix([[pm[i+4*j]for j in range(4)]for i in range(4)]);inv=pm.inverted()
def load(n,c=(.25,.55,.58,1),local=False,source=None):
 bpy.ops.wm.stl_import(filepath=str((source or v)/(n+'.stl')));o=bpy.context.object;o.name=n;o.color=c
 if local:
  for p in o.data.vertices:p.co=inv@p.co
 return o
def render(name,pos,target,scale):
 cam.location=pos;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;s.render.filepath=str(v/(name+'.png'));bpy.ops.render.render(write_still=True)
tray=load('drop-projector-tray-t5',local=True)
for i in range(4):
 c=load(f'tray-nut-cover-{i}',(.95,.6,.2,1),True);c.location.y=12
 load(f'drop-projector-tray-t5-frame-nut-{i}',(.65,.7,.73,1),True)
render('tray-top-load',(170,230,150),(67.5,-2,37.5),190)
for o in [o for o in bpy.data.objects if o.type=='MESH']:bpy.data.objects.remove(o,do_unlink=True)
load('tray-nut-fit-coupon');cap=load('tray-nut-cover-local',(.95,.6,.2,1));cap.location.y=8
nut=load('drop-projector-tray-t5-frame-nut-0',(.65,.7,.73,1),True)
for p in nut.data.vertices:p.co.x-=12;p.co.z-=7
render('tray-pocket-detail',(25,40,30),(0,-2,2),37)
for o in [o for o in bpy.data.objects if o.type=='MESH']:bpy.data.objects.remove(o,do_unlink=True)
head=load('head',(.79,.83,.81,1));cradle=load('mirror-removable-cradle',(.95,.6,.2,1))
md=json.load(open(v/'head-check.json'))['mirrorMatrix'];di=Vector(md[8:11]);cradle.location=di*20+Vector((100,0,0))
for n in ['mirror-glass','printed-mirror-backing','printed-mirror-fixed-plate']:
 ob=load(n,(.5,.65,.7,1),source=r/'assembly');ob.location=di*20+Vector((100,0,0))
render('mirror-cartridge',(460,-340,220),(30,0,175),400)
for o in [o for o in bpy.data.objects if o.type=='MESH' and o.name!='head']:o.hide_render=True
for i in range(4):
 ob=load(f'mirror-cartridge-bolt-{i}',(.4,.45,.47,1));ob.location=-di*20
render('rear-access',(-450,-330,460),(-40,0,192),310)
