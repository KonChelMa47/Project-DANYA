exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'unified-head-v5';o=loadpath(out/'magnet-carrier-before-face-fit.stl');tool=loadpath(out/'face.stl');tool.data.update()
norms=[v.normal.copy()for v in tool.data.vertices]
for v,n in zip(tool.data.vertices,norms):v.co+=n*.12
bpy.context.view_layer.objects.active=o;m=o.modifiers.new('face adhesive clearance','BOOLEAN');m.operation='DIFFERENCE';m.solver='MANIFOLD';m.object=tool;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(tool,do_unlink=True);print(export(o,'magnet-carrier-face-fit'),flush=True)
