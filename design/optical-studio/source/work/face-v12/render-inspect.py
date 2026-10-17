import bpy,math
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v12'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
for name in ['face','head-upper','head-lower','magnet-carrier']:
 bpy.ops.wm.stl_import(filepath=str((r/'assembly' if name=='face'else out)/((name if name=='face'else name+'-covered')+'.stl')));o=bpy.context.object;mat=bpy.data.materials.new(name);mat.diffuse_color=((.05,.4,.8,1) if name=='magnet-carrier' else (.78,.79,.77,1));o.data.materials.append(mat)
s=bpy.context.scene;bpy.ops.object.camera_add(location=(500,-280,220));cam=bpy.context.object;cam.rotation_euler=(Vector((5,0,162))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=255;cam.data.clip_end=3000;s.camera=cam;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.08,.10,.12);s.render.resolution_x=1100;s.render.resolution_y=1100;s.render.resolution_percentage=100;s.render.filepath=str(out/'carrier-inspection.png');bpy.ops.render.render(write_still=True)
