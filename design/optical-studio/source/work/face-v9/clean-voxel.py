import bpy,bmesh,json,time
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
obs=[]
for n in ['face-local-patched']+[f'face-holder-{i}'for i in range(4)]:
 bpy.ops.wm.stl_import(filepath=str(out/(n+'.stl')));obs.append(bpy.context.object)
bpy.ops.object.select_all(action='DESELECT')
for o in obs:o.select_set(True)
bpy.context.view_layer.objects.active=obs[0];bpy.ops.object.join();face=bpy.context.object
face.data.remesh_voxel_size=.15;face.data.remesh_voxel_adaptivity=.10;face.data.use_remesh_preserve_volume=False;face.data.use_remesh_fix_poles=False;bpy.ops.object.voxel_remesh();print('voxel joined',len(face.data.vertices),flush=True);print(export(face,'face-clean-voxel'),flush=True)
