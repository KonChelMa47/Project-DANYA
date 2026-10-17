exec(open('work/face-v11/redesign.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=r/'face-v12'
def cylinder_at(c,radius,depth):
 bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=radius,depth=depth,location=c,rotation=(0,math.pi/2,0));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
def intersect(o,tool):
 bpy.context.view_layer.objects.active=o;m=o.modifiers.new('inside original head envelope','BOOLEAN');m.operation='INTERSECT';m.solver='MANIFOLD';m.object=tool;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(tool,do_unlink=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
o=load(r/'archive-before-face-v11/assembly/magnet-carrier.stl');h=load(r/'archive-before-face-v11/assembly/head-upper.stl');lo=load(r/'archive-before-face-v11/assembly/head-lower.stl')
cut(o,box((-30,0,280),(140,180,80)));cut(o,box((0,0,20),(250,200,69))) # no tabs, crop below z54.5
for i,y in [(0,-18.054),(1,18.054)]:
 z=255.0
 front=cylinder_at((42.75,y,z),2.1,2);intersect(front,load(out/'head-envelope.stl'));union(o,front)
 rear=cylinder_at((39.9,y,z),2.1,3);intersect(rear,load(out/'head-envelope.stl'));union(h,rear)
 # Trim the old return lips that would interlock with the smaller head-side seat.
 cut(o,box((31.75,y,260),(20,7.0,20)))
 cut(o,cylinder_at((42.3,y,z),1.65,1.4));cut(h,cylinder_at((41.05,y,z),1.65,1.5))
 for side,x in [('face',42.3),('head',41.0)]:
  mag=cylinder_at((x,y,z),1.5,1);export(mag,f'magnet-clear-{i}-{side}-magnet');bpy.data.objects.remove(mag,do_unlink=True)
# Stay inside the unmodified head+face exterior; remove high housing corners.
intersect(o,load(out/'head-envelope.stl'))
# Optical trim restricted to the front upper coupling region.
for obj in [o,h]:
 cavity=load(r/'face-v10/inner-cavity.stl');cavity.location.z=.25;intersect(cavity,box((57,0,270),(40,70,60)));cut(obj,cavity)
# Internal finger access at the underside, without any protruding handle.
# Existing neck opening provides concealed access; preserve the outer neck.
reports={}
for name,obj in [('magnet-carrier',o),('head-upper',h),('head-lower',lo)]:reports[name]=export(obj,name);print(name,reports[name],flush=True)
(out/'geometry.json').write_text(json.dumps(reports,indent=2))
