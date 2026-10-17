exec(open('work/face-v10/build.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
from mathutils import Vector
import math
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
objects={n:load(out/(source+'.stl'))for n,source in [('magnet-carrier','magnet-carrier'),('head-upper','head-upper-fixed'),('head-lower','head-lower-fixed')]}
for i in range(4):
 for side in ['face','head']:
  magnet=load(out/f'magnet-clear-{i}-{side}-magnet.stl');coords=np.array([v.co[:]for v in magnet.data.vertices]);center=(coords.min(0)+coords.max(0))/2;center[0]+= -.1 if side=='face'else .1;bpy.data.objects.remove(magnet,do_unlink=True)
  bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=3.10,depth=3.30,location=center,rotation=(0,math.pi/2,0));tool=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);o=objects['magnet-carrier'if side=='face'else'head-upper'if i<2 else'head-lower'];bpy.context.view_layer.objects.active=o;m=o.modifiers.new('restore magnet seating cavity','BOOLEAN');m.operation='DIFFERENCE';m.solver='MANIFOLD';m.object=tool;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(tool,do_unlink=True)
reports={}
for name,o in objects.items():reports[name]=export(o,name+'-pockets');print(name,reports[name],flush=True)
(out/'pocket-geometry.json').write_text(json.dumps(reports,indent=2))
