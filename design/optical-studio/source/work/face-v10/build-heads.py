exec(open('work/face-v10/build.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
def scale(o):
 for v in o.data.vertices:v.co.y*=1.003;v.co.z=160+(v.co.z-160)*1.003
 o.data.update();return o
def boolean(a,b,op,remove=False):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
carrier=scale(load(out/'carrier-finished.stl'));reports={'carrier':export(carrier,'magnet-carrier')};base=load(r/'archive-before-integrated-magnets/assembly/head.stl');heads={n:load(r/'face-v9'/(n+'-finished.stl'))for n in ['head-upper','head-lower']}
positions=json.load(open(r/'face-v9/positions.json'))
for row in positions:
 i=row['id'];h=heads['head-upper'if row['top']else'head-lower'];old=load(r/'face-v9'/f'head-holder-{i}.stl');boolean(old,base,'DIFFERENCE');boolean(h,old,'DIFFERENCE',True);new=scale(load(r/'face-v9'/f'head-holder-{i}.stl'));boolean(h,new,'UNION',True);relief=scale(load(r/'face-v9'/f'head-relief-{i}.stl'));boolean(h,relief,'DIFFERENCE',True)
 for side in ['head','face']:
  name=f'magnet-clear-{i}-{side}-magnet';m=load(r/'face-v9'/(name+'.stl'));center=np.array(row['center']);delta=np.array([0,center[1]*.003,(center[2]-160)*.003]);m.location=delta.tolist();bpy.context.view_layer.objects.active=m;bpy.ops.object.transform_apply(location=True,rotation=False,scale=False);export(m,name)
 print('pair',i,flush=True)
for name,h in heads.items():
 for delta in [(0,0,0),(.2,0,0),(-.2,0,0),(0,.2,0),(0,-.2,0),(0,0,.2),(0,0,-.2)]:
  carrier.location=delta;boolean(h,carrier,'DIFFERENCE')
 carrier.location=(0,0,0);reports[name]=export(h,name);print(name,reports[name],flush=True)
(out/'head-build-report.json').write_text(json.dumps(reports,indent=2))
