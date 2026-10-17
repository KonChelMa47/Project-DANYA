import bpy,json,math
from pathlib import Path
from mathutils import Vector,Matrix
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'split-head-v6';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.93,.95,.95);s.view_settings.view_transform='Standard';s.render.resolution_x=1400;s.render.resolution_y=1100;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=2000;s.camera=cam
objs={}
def load(n,col,folder=v):
 bpy.ops.wm.stl_import(filepath=str(folder/(n+'.stl')));o=bpy.context.object;o.name=n;o.color=col;objs[n]=o;return o
def render(n,cp,target,scale):
 cam.location=cp;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;s.render.filepath=str(v/(n+'.png'));bpy.ops.render.render(write_still=True)
upper=load('head-upper',(.80,.86,.82,1));lower=load('head-lower',(.62,.75,.73,1))
for i in range(4):
 for typ in ['bolt','washer','nut']:load(f'head-seam-{typ}-{i}',(.4,.46,.49,1))
for i in range(2):load(f'head-alignment-pin-{i}',(.95,.6,.2,1))
upper.location.z=85
for n,o in objs.items():
 if n.startswith('head-seam-bolt')or n.startswith('head-seam-washer'):o.location.z=105
render('split-overview',(400,-420,280),(-25,0,205),440)
# Inside closeup shows load-carrying bosses and ribs, not just shell edges.
upper.location.z=22
for n,o in objs.items():
 if n.startswith('head-seam-bolt')or n.startswith('head-seam-washer'):o.location.z=40
p=json.load(open(v/'head-split-check.json'))['flanges'][1];c=Vector(p['center']);d=Vector(p['radial']);render('joint-detail',c-d*85+Vector((0,0,48)),c+Vector((0,0,12)),95)
for o in objs.values():o.location=(0,0,0)
render('assembled-rear',(-430,-360,260),(-30,0,164),300)
# Exactly the supplied print orientations, with the two cut faces on the bed.
for n,o in objs.items():o.hide_render=n not in ['head-upper','head-lower']
for n,o,flip,xx in [('head-upper',upper,False,-100),('head-lower',lower,True,100)]:
 M=Matrix.Rotation(math.pi,4,'X')if flip else Matrix.Identity(4)
 for vv in o.data.vertices:vv.co=M@vv.co
 lo=Vector([min(vv.co[i]for vv in o.data.vertices)for i in range(3)]);hi=Vector([max(vv.co[i]for vv in o.data.vertices)for i in range(3)]);offset=Vector((xx-(lo.x+hi.x)/2,-(lo.y+hi.y)/2,-lo.z))
 for vv in o.data.vertices:vv.co+=offset
render('print-orientation',(380,-500,400),(0,0,45),475)
