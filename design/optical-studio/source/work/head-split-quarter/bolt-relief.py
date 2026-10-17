exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'head-split-quarter';data=json.load(open(out/'head-split-check.json'));reports={}
def cut(a,b):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;m=a.modifiers.new('clearance','BOOLEAN');m.operation='DIFFERENCE';m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
for name in ['head-upper','head-lower']:
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/(name+'.stl')));head=bpy.context.object
 for p in data['flanges']:
  x,y,z=p['boltCenter'];bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=2.2,depth=60,location=(x,y,106.7));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);cut(head,o)
  if name=='head-upper':
   bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=4.7,depth=23.3,location=(x,y,128.35));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);cut(head,o)
 rep=export(head,name);reports[name]=rep;print(name,rep,flush=True)
(out/'bolt-relief-report.json').write_text(json.dumps(reports,indent=2))
