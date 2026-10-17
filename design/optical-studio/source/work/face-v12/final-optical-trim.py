exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
o=load(out/'head-upper-covered.stl');clean(o)
# Trim only the underside of the two new magnet seats against the nominal light envelope.
cavity=load(r/'face-v10/inner-cavity.stl');clean(cavity);cavity.location.z=.3
bpy.context.view_layer.objects.active=cavity;bpy.ops.object.transform_apply(location=True,rotation=False,scale=False)
intersect(cavity,box((39.5,0,250),(8,50,25)));clean(cavity);cut(o,cavity);print(export(o,'head-upper-covered'),flush=True)
