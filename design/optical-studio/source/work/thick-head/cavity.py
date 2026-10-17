exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
def exact(a,b,mode):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='EXACT';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
cavity=loadpath(src/'face-v10/inner-cavity.stl')
for n in ['head-upper','head-lower']:exact(cavity,loadpath(src/'solid-seam-v2'/(n+'.stl')),'DIFFERENCE')
print(export(cavity,'optical-relief'),flush=True)
