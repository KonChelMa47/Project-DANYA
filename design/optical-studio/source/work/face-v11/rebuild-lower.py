exec(open('work/face-v11/redesign.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
h=load(r/'assembly/head-lower.stl');cut(h,box((44,0,46),(40,80,26)))
print(export(h,'head-lower-release'),flush=True)
