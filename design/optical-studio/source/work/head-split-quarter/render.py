import bpy,json
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/head-split-quarter';bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;objects=[]
for name,col in [('head-upper',(.27,.60,.66,1)),('head-lower',(.87,.60,.33,1))]:
 bpy.ops.wm.stl_import(filepath=str(r/(name+'.stl')));o=bpy.context.object;o.name=name;o.color=col;objects.append(o)
s.world=bpy.data.worlds.new('World');s.world.color=(.94,.96,.97);s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.render.resolution_x=1200;s.render.resolution_y=1100;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.view_settings.view_transform='Standard';s.view_settings.look='None'
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=3000;s.camera=cam
for name,loc,target,scale,up in [('assembled',(350,-580,260),(-25,0,165),290,0),('split-view',(430,-620,320),(-25,0,185),365,35),('joint-detail',(140,-360,205),(-32,-52,119),95,17)]:
 objects[0].location.z=up;cam.location=loc;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;s.render.filepath=str(r/(name+'.png'));bpy.ops.render.render(write_still=True)
objects[0].location.z=0;bpy.ops.wm.save_as_mainfile(filepath=str(r/'head-split-quarter.blend'))
