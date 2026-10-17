exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'head-split-quarter';out.mkdir(exist_ok=True);bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
objs=[]
for n in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(src/'assembly'/(n+'.stl')));o=bpy.context.object;o.name=n;clean(o);objs.append(o)
m=objs[0].modifiers.new('join','BOOLEAN');m.operation='UNION';m.solver='MANIFOLD';m.object=objs[1];bpy.context.view_layer.objects.active=objs[0];bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(objs[1],do_unlink=True);head=objs[0];print('UNION',export(head,'head-rejoined-clean'),flush=True)
