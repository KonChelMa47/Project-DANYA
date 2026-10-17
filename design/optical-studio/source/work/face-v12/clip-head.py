exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
o=load(out/'head-upper-covered.stl');env=load(out/'head-envelope.stl');clean(o);clean(env)
bpy.context.view_layer.objects.active=o;m=o.modifiers.new('outer form limit','BOOLEAN');m.operation='INTERSECT';m.solver='EXACT';m.use_self=True;m.object=env;bpy.ops.object.modifier_apply(modifier=m.name);print(export(o,'head-upper-covered'),flush=True)
