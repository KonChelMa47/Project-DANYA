exec(open('work/face-v10/build.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
import math
out=r/'face-v11'
def box(center,size):
 bpy.ops.mesh.primitive_cube_add(size=1,location=center);o=bpy.context.object;o.dimensions=size;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
def cut(a,b):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new('clearance','BOOLEAN');m.operation='DIFFERENCE';m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def cyl(center,depth=3.3):
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=3.1,depth=depth,location=center,rotation=(0,math.pi/2,0));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
o=load(out/'magnet-carrier.stl');cut(o,box((-30,0,280),(140,180,80))) # x<40,z>240; clears old upper return arms
h=load(r/'assembly/head-upper.stl')
for i,y in [(0,-18.054),(1,18.054)]:
 # Forward-facing magnet seats ahead of the mirror; short, rigid bridges into the rim.
 z=255.5
 union(o,box((51.3,y,z),(6,10,10)))
 union(o,box((53.2,y,251),(4,8,8)))

 cut(o,cyl((49.85,y,z)))
 union(h,box((43.2,y,z),(10,10,10)))
 cut(h,cyl((46.6,y,z),3.6))
 for side,x in [('face',49.85),('head',46.45)]:
  m=load(r/'assembly'/f'magnet-clear-{i}-{side}-magnet.stl');coords=np.array([v.co[:]for v in m.data.vertices]);old=(coords.min(0)+coords.max(0))/2
  for v in m.data.vertices:v.co+=__import__('mathutils').Vector(np.array([x,y,z])-old)
  export(m,f'magnet-clear-{i}-{side}-magnet');bpy.data.objects.remove(m,do_unlink=True)
# Trim only the added supports against the already established optical envelope.
cavity=load(r/'face-v10/inner-cavity.stl');cavity.location.z=.6;cut(o,cavity)
reports={'magnet-carrier':export(o,'magnet-carrier-revised'),'head-upper':export(h,'head-upper')};print(reports,flush=True);(out/'revised-geometry.json').write_text(json.dumps(reports,indent=2))
