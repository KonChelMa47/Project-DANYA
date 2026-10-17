exec(open('work/assembly-integrity/rebuild.py').read().split("p=np.load(src/")[0])
bpy.ops.wm.stl_import(filepath=str(out/'head-lower.stl'));head=bpy.context.object;op(head,box((0,0,-74.675),(600,600,250.65)),'DIFFERENCE');print(export(head,'head-lower'))
