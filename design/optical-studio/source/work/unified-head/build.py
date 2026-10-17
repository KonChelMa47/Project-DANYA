exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'unified-head-v5'
def direct(a,b,mode,remove=True):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
def cube(lo,hi):
 bpy.ops.mesh.primitive_cube_add(size=1,location=[(lo[i]+hi[i])/2 for i in range(3)]);o=bpy.context.object;o.dimensions=[hi[i]-lo[i]for i in range(3)];bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
outer=loadpath(out/'inset-envelope.stl');shell=loadpath(out/'smooth-shell.stl');direct(shell,cube([-500,-500,-500],[500,500,50.65]),'DIFFERENCE')
# Avoid a newly closed neck floor: keep only the region outside the existing neck opening.
# A generous cylinder follows the neck hollow and joins the original cut rim.
for name in ['head-upper','head-lower']:
 head=loadpath(src/'thick-head-v3'/(name+'.stl'));backup=head.copy();backup.data=head.data.copy();bpy.context.collection.objects.link(backup)
 direct(head,outer,'INTERSECT',False)
 # Preserve the functional face-facing magnetic receivers beyond the seam envelope.
 bpy.data.objects.remove(backup,do_unlink=True);backup=None
 for center,radius,depth in ([( (40,-18.054,255),2.1,5),((40,18.054,255),2.1,5)]if name=='head-upper'else [((18,-30.09,65.70844),4.4,7),((18.2,30.09,64.76681),4.4,7)]):
  orig=loadpath(src/'thick-head-v3'/(name+'.stl'));bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=radius,depth=depth,location=center,rotation=(0,math.pi/2,0));cy=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);direct(orig,cy,'INTERSECT')
  if backup is None:backup=orig
  else:direct(backup,orig,'UNION')
 portion=shell.copy();portion.data=shell.data.copy();bpy.context.collection.objects.link(portion)
 direct(portion,cube([-500,-500,-500],[500,500,165])if name=='head-upper'else cube([-500,-500,165],[500,500,500]),'DIFFERENCE');direct(head,portion,'UNION');direct(head,backup,'UNION');print(name,export(head,name),flush=True)
