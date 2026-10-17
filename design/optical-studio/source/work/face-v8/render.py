import bpy,math
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';o=r/'face-v8'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
for name,color in [('face',(.63,.78,.78,1))]:
 bpy.ops.wm.stl_import(filepath=str(o/(name+'.stl')));ob=bpy.context.object;mat=bpy.data.materials.new(name);mat.diffuse_color=color;ob.data.materials.append(mat)
s=bpy.context.scene;bpy.ops.object.camera_add(location=(-430,-260,235));cam=bpy.context.object;cam.rotation_euler=(Vector((20,0,163))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=240;cam.data.clip_end=3000;s.camera=cam;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.06,.08,.1);s.render.resolution_x=1100;s.render.resolution_y=1050;s.render.resolution_percentage=100;s.render.filepath=str(o/'reinforced-face.png');bpy.ops.render.render(write_still=True)
