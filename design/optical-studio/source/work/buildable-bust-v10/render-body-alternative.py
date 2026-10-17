import bpy,numpy as np,json
from pathlib import Path
from mathutils import Vector
r=Path('outputs/danya-simulator/assets');bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def mesh(name,prefix,color):
 v=np.fromfile(str(prefix)+'.positions.bin',np.float32).reshape(-1,3);f=np.fromfile(str(prefix)+'.indices.bin',np.uint32).reshape(-1,3);m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);o.color=color
 for p in m.polygons:p.use_smooth=True
 return o
head=mesh('head',r/'buildable-bust-v10/head-inner-relief',(.72,.76,.75,1));head.location.z=8
face=mesh('face',r/'torso-flat-study/face',(.84,.79,.71,1));face.location.z=8
body=mesh('body',r/'buildable-bust-v10/body-wall3',(.56,.65,.65,1))
collar=mesh('collar',r/'buildable-bust-v3/collar',(.24,.28,.3,1));original=np.array([v.co[:]for v in collar.data.vertices]);top=json.load(open(r/'buildable-bust-v2/neck-datum.json'))['sectionZ']
cam=bpy.data.cameras.new('camera');o=bpy.data.objects.new('camera',cam);bpy.context.collection.objects.link(o);o.location=(650,-440,280);target=Vector((0,0,65));o.rotation_euler=(target-o.location).to_track_quat('-Z','Y').to_euler();cam.type='ORTHO';cam.ortho_scale=535;bpy.context.scene.camera=o
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_shadows=True;s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.background_type='WORLD';s.world.color=(.045,.055,.065);s.render.resolution_x=900;s.render.resolution_y=1050;s.render.resolution_percentage=100
for alt in [0,1]:
 if alt:body.hide_render=True;body2=mesh('alternative',r/'buildable-bust-v10/body-clearance-alt1',(.56,.65,.65,1))
 for i,p in enumerate(original):
  u=(p[2]-24)/(top-24);t=-np.pi+(i%192)*2*np.pi/192;collar.data.vertices[i].co=(p[0]+(9+6*alt)*np.cos(t)*(1-u),p[1]+(10+8*alt)*np.sin(t)*(1-u),p[2]+8*u)
 s.render.filepath=str(Path('work/buildable-bust-v10')/('body-review-'+str(alt)+'.png'));bpy.ops.render.render(write_still=True)
