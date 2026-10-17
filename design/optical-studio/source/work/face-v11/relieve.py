exec(open('work/face-v11/redesign.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
o=load(out/'magnet-carrier-revised.stl');heads={'head-upper':load(out/'head-upper.stl'),'head-lower':load(r/'assembly/head-lower.stl')}
# Clear internal undercuts along the first 25 mm of straight forward withdrawal.
# Small offsets add tolerance around the sliding surfaces.
for name,h in heads.items():
 if name=='head-upper':
  cavity=load(r/'face-v10/inner-cavity.stl');cavity.location.z=.6
  clip=box((57,0,270),(40,70,60));bpy.context.view_layer.objects.active=cavity;m=cavity.modifiers.new('local upper supports only','BOOLEAN');m.operation='INTERSECT';m.solver='MANIFOLD';m.object=clip;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(clip,do_unlink=True);cut(h,cavity)
 for x in [-.4,.1,.6,1.1,1.6,2.1,2.6,3.1,3.6,4.1,5.1,7.1,10.1]:
  o.location=(float(x),0,0)
  bpy.context.view_layer.objects.active=h;m=h.modifiers.new('forward removal clearance','BOOLEAN');m.operation='DIFFERENCE';m.solver='MANIFOLD';m.object=o;bpy.ops.object.modifier_apply(modifier=m.name)
 o.location=(0,0,0)
 print(name,export(h,name+'-release'),flush=True)
