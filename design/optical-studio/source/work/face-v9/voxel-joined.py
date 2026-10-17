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
face.data.remesh_voxel_size=.15;face.data.remesh_voxel_adaptivity=.10;bpy.ops.object.voxel_remesh();print('voxel joined',len(face.data.vertices),flush=True);print(export(face,'face-voxel-joined'),flush=True)
# Recover the original optical screen and its minimum thickness without projecting vertices.
bpy.ops.wm.stl_import(filepath=str(r/'face-v8/screen-after.stl'));o=bpy.context.object;bpy.context.view_layer.objects.active=face;m=face.modifiers.new('Preserve optical screen','BOOLEAN');m.operation='UNION';m.solver='MANIFOLD';m.object=o;bpy.ops.object.modifier_apply(modifier=m.name);assert not face.modifiers;bpy.data.objects.remove(o,do_unlink=True)
print(export(face,'face-voxel-screen'),flush=True)
