import bpy,json
from mathutils import Vector
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'guide';bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'));s=bpy.context.scene
s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.92,.94,.95);s.render.resolution_x=1400;s.render.resolution_y=1000;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.view_settings.view_transform='Standard';s.view_settings.look='None'
meshes={o.name:o for o in bpy.data.objects if o.type=='MESH'};orig={n:o.location.copy()for n,o in meshes.items()}
for n,o in meshes.items():o.color=(.3,.57,.59,1)if n not in ['head','face','projector']else((.79,.83,.81,1)if n!='projector'else(.7,.34,.25,1))
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=3000;s.camera=cam
base=[n for n in meshes if n.startswith('base-')];frame=base+[n for n in meshes if n.startswith(('head-column','projector-stand','lower-mirror-stand'))];tray=[n for n in meshes if n.startswith(('drop-projector-tray-t5','tray-nut-cover-'))]+['projector']+[n for n in meshes if n.startswith('projector-stand')]
configs=[('overview',[n for n in meshes if n!='head'],(650,-850,400),(0,0,40),680,{}),('base',base,(450,-650,-470),(0,0,-170),390,{n:(0,0,-28)for n in base if n.startswith(('base-foot','base-join'))}),('frame',frame,(650,-850,400),(0,0,-65),500,{}),('projector',tray,(520,-600,240),(0,0,-75),350,{'projector':(0,0,45)}),('head', ['head-upper','head-lower','head-arm-left','head-arm-right']+[n for n in meshes if n.startswith('head-connect-')],(360,-260,140),(-30,0,145),430,{}),('face', ['head-upper','head-lower','face']+[n for n in meshes if n.startswith('magnet-clear-')],(500,-500,220),(25,0,165),360,{'face':(80,0,0),**{n:(80,0,0)for n in meshes if 'face-magnet'in n}})]
for name,visible,cp,target,scale,offsets in configs:
 for n,o in meshes.items():o.hide_render=n not in visible or 'coupon'in n or 'master'in n;o.location=orig[n]+Vector(offsets.get(n,(0,0,0)))
 cam.location=cp;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;s.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
print('Guide renders complete')
