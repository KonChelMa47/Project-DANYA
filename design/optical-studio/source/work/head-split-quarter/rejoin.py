exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'head-split-quarter';out.mkdir(exist_ok=True);bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
objs=[]
for n in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(src/'assembly'/(n+'.stl')));o=bpy.context.object;o.name=n;clean(o);objs.append(o)
boolean(objs[0],objs[1],'UNION');head=objs[0];print('UNION',export(head,'head-rejoined'),flush=True)
# Close former blind alignment pockets; retain the existing internal ribs, preserving the original light clearance.
p=json.load(open(src/'split-head-v6/head-split-check.json'))
for pin in p['pins']:
 c=Vector(pin['center']);bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=2.2,depth=6.5,location=(c.x,c.y,165));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(head,o,'UNION')
 # Second pocket was elongated along the two-pin axis, reproduce fill.
 if pin['id']==1:
  axis=(Vector(p['pins'][1]['center'])-Vector(p['pins'][0]['center'])).normalized()
  for off in [-.4,.4]:
   cc=c+axis*off;bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=2.2,depth=3.3,location=(cc.x,cc.y,166.55));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(head,o,'UNION')
rep=export(head,'head-rejoined');(out/'rejoin-report.json').write_text(json.dumps(rep,indent=2));assert rep['components']==1 and rep['nonManifoldEdges']==0,rep
bpy.ops.wm.save_as_mainfile(filepath=str(out/'head-rejoined.blend'));print('DONE',rep,flush=True)
