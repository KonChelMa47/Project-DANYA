exec(open('work/unified-head/carrier.py').read().split('# Clip old supports')[0])
def cut(a,b,mode):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def box(c,d):
 bpy.ops.mesh.primitive_cube_add(size=1,location=c);o=bpy.context.object;o.dimensions=d;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
for num,(c,d)in enumerate([((42,-18.054,257),(8,10,12)),((42,18.054,257),(8,10,12)),((23,-30.09,60),(12,20,24)),((23,30.09,60),(12,20,24))]):
 old=loadpath(src/'solid-seam-v2/magnet-carrier.stl');cut(old,box(c,d),'INTERSECT');cut(old,loadpath(out/'carrier-inset-envelope.stl'),'INTERSECT');print('support',num,export(old,'support-test-'+str(num)),flush=True);direct(band,old,'UNION');print('union',num,len(band.data.polygons),flush=True)
print(export(band,'magnet-carrier-localized'),flush=True)
