exec(open('work/assembly-integrity/rebuild.py').read().split("p=np.load(src/")[0])
import math
bpy.ops.wm.stl_import(filepath=str(out/'head-lower.stl'));head=bpy.context.object;n=Vector((-.45,0,1)).normalized();top=Vector((0,0,48.4));bpy.ops.mesh.primitive_cube_add(size=1,location=top-n*300);tool=bpy.context.object;tool.rotation_euler[1]=-math.atan(.45);tool.dimensions=(1000,1000,600);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);op(head,tool,'DIFFERENCE');print(export(head,'head-lower'))
