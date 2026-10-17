exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'split-head-v6';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(out/'head.stl'));head=bpy.context.object;clean(head);print('HEAD',len(head.data.vertices),head.matrix_world,flush=True)
bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,-167.5));cube=bpy.context.object;cube.dimensions=(1000,1000,665);bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(head,cube,'DIFFERENCE');print('UPPER',len(head.data.vertices),flush=True);export(head,'debug-upper')
