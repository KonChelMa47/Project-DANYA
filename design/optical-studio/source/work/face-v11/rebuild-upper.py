exec(open('work/face-v11/redesign.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
h=load(r/'assembly/head-upper.stl')
for y in [-18.054,18.054]:union(h,box((43.1,y,255.5),(9.8,10,10)))
cavity=load(r/'face-v10/inner-cavity.stl');cavity.location.z=.6;clip=box((57,0,270),(40,70,60));bpy.context.view_layer.objects.active=cavity;m=cavity.modifiers.new('local optical clearance','BOOLEAN');m.operation='INTERSECT';m.solver='MANIFOLD';m.object=clip;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(clip,do_unlink=True);cut(h,cavity)
for y in [-18.054,18.054]:
 cut(h,cyl((46.6,y,255.5),3.6))
 cut(h,box((54,y,261),(12,20,20)))
print(export(h,'head-upper-release'),flush=True)
