exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'head-split-quarter';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/'head-upper.stl'));head=bpy.context.object
for n in ['bridge-125-2','bridge-155-4','bridge-164-6']:
 bpy.ops.wm.stl_import(filepath=str(out/(n+'.stl')));b=bpy.context.object;clean(head);clean(b);bpy.context.view_layer.objects.active=head;m=head.modifiers.new('bridge','BOOLEAN');m.operation='UNION';m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
rep=export(head,'head-upper');(out/'bridge-report.json').write_text(json.dumps(rep,indent=2));print('BRIDGED',rep,flush=True)
