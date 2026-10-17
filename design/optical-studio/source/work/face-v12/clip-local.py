exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
def boolean(a,b,op):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;m=a.modifiers.new('local exterior trim','BOOLEAN');m.operation=op;m.solver='EXACT';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
o=load(out/'head-upper-covered.stl');clean(o)
for y in [-18.054,18.054]:
 tool=box((40,y,259),(3.2,4.2,5));env=load(out/'head-envelope.stl');clean(env);boolean(tool,env,'DIFFERENCE');boolean(o,tool,'DIFFERENCE')
print(export(o,'head-upper-covered'),flush=True)
