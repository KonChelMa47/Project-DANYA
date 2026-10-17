exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'head-split-quarter';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def op(a,b,mode,remove=True):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
bpy.ops.wm.stl_import(filepath=str(out/'head-upper.stl'));head=bpy.context.object;head.name='head-upper'
bpy.ops.wm.stl_import(filepath=str(src/'archive-before-printable-mounts-v5/assembly/head.stl'));original=bpy.context.object;original.name='protected-original-shell-and-mounts'
for deg in [103,123,240,257]:
 bpy.ops.wm.stl_import(filepath=str(src/'split-head-v6'/f'flange-{deg}.stl'));cut=bpy.context.object
 # Enclose tiny Boolean-rounding remnants; protect every volume of the pre-seam head.
 center=sum((v.co for v in cut.data.vertices),Vector())/len(cut.data.vertices)
 for v in cut.data.vertices:v.co=center+(v.co-center)*1.015
 op(cut,original,'DIFFERENCE',False);op(head,cut,'DIFFERENCE');print('REMOVED OLD',deg,flush=True)
rep=export(head,'head-upper');(out/'old-joints-removal.json').write_text(json.dumps({'headUpper':rep,'removedAngles':[103,123,240,257],'method':'Subtract enlarged former flange volumes minus protected pre-seam original head. Preserve original shell and integrated mounts.'},indent=2));print('FINAL',rep,flush=True)
