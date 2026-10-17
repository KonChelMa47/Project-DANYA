exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
for n in ['head-upper','head-lower']:
 head=loadpath(out/(n+'-clean.stl'));tool=loadpath(out/'optical-relief.stl');bpy.context.view_layer.objects.active=head;m=head.modifiers.new('preserve original optical cavity','BOOLEAN');m.operation='DIFFERENCE';m.solver='EXACT';m.object=tool;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(tool,do_unlink=True);print(n,export(head,n),flush=True);bpy.data.objects.remove(head,do_unlink=True)
