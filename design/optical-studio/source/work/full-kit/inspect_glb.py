import bpy,json,numpy as np
from pathlib import Path
from mathutils import Vector
p=Path(__file__).resolve().parent
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.import_scene.gltf(filepath='/Users/daniilmalchenko/Downloads/projector module 3d model.glb')
objs=[o for o in bpy.context.scene.objects if o.type=='MESH'];pts=[o.matrix_world@v.co for o in objs for v in o.data.vertices];a=np.array(pts);lo=a.min(0);hi=a.max(0);center=Vector((lo+hi)/2);size=float((hi-lo).max());print(json.dumps({'objects':[{'name':o.name,'vertices':len(o.data.vertices),'dimensions':list(o.dimensions)} for o in objs],'bounds':[lo.tolist(),hi.tolist()],'size':(hi-lo).tolist()}),flush=True)
bpy.context.scene.render.engine='BLENDER_EEVEE_NEXT';bpy.context.scene.render.resolution_x=800;bpy.context.scene.render.resolution_y=800;bpy.context.scene.render.resolution_percentage=100;bpy.context.scene.world.color=(.3,.3,.3)
for v in [(2,-3,4),(-3,2,4),(1,3,-2)]:
 bpy.ops.object.light_add(type='AREA',location=center+Vector(v)*size);bpy.context.object.data.energy=1500*size*size;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=size*3;bpy.context.object.rotation_euler=(center-bpy.context.object.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add();cam=bpy.context.object;bpy.context.scene.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=size*1.5;cam.data.clip_start=size*.001;cam.data.clip_end=size*100
for i,d in enumerate([(2,-3,3),(-2,3,3),(0,0,-4)]):
 cam.location=center+Vector(d)*size;cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();bpy.context.scene.render.filepath=str(p/f'glb-view-{i}.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(p/'imported-projector.blend'))
