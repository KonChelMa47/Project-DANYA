exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
def boolean(a,b,op,remove=True):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;mod=a.modifiers.new(op,'BOOLEAN');mod.operation=op;mod.solver='MANIFOLD';mod.object=b;bpy.ops.object.modifier_apply(modifier=mod.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
out=root/'outputs/danya-simulator/assets/projection-prototype/head-split-quarter';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(out/'head-rejoined-repaired.stl'));head=bpy.context.object;head.name='head';clean(head)
rows=json.load(open(out/'candidates.json'));base=[next(p for p in rows if p['angle']==d)for d in [103,257]];selected=[];Z=106.7
for p in base:
 bpy.ops.wm.stl_import(filepath=str(out/f"flange-{p['angle']}.stl"));boolean(head,bpy.context.object,'UNION')
 for off in [-7.5,7.5]:
  q=dict(p);q['center']=list(Vector(p['center'])+Vector(p['tangent'])*(off+3.5));selected.append(q)
pins=[{'id':i,'center':p['center'],'upperSlot':i==1}for i,p in enumerate(base)]
for i,p in enumerate(selected):
 c=Vector(p['center']);t=Vector(p['tangent']);s=c-t*3.5;p['boltCenter']=list(s);p['name']=f'head-seam-{i}'
 bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=4.7,depth=.4,location=(s.x,s.y,116.9));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(head,o,'DIFFERENCE')
 # Continuous clearance; nut wells on the upper shell, with a load-bearing floor.
 bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=2.2,depth=18.2,location=(s.x,s.y,107.7));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(head,o,'DIFFERENCE')
 bpy.ops.mesh.primitive_cylinder_add(vertices=6,radius=7.4/math.sqrt(3),depth=3.6,location=(s.x,s.y,100.4),rotation=(0,0,math.pi/6));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(head,o,'DIFFERENCE')

 print('FLANGE',i,flush=True)
# Open an inward, bottom-open loading channel so each nut can slide in without crossing the curved shell.
# The4.5mm shoulder above the nut remains intact and carries clamp load.
def hull2(points):
 points=sorted(set(points))
 def cross(a,b,c):return(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
 low=[]
 for q in points:
  while len(low)>=2 and cross(low[-2],low[-1],q)<=0:low.pop()
  low.append(q)
 high=[]
 for q in reversed(points):
  while len(high)>=2 and cross(high[-2],high[-1],q)<=0:high.pop()
  high.append(q)
 return low[:-1]+high[:-1]
for p in selected:
 c=Vector(p['boltCenter']);d=Vector(p['radial']);pts=[]
 for offset in [0,20]:
  for j in range(6):
   t=j*math.tau/6;pts.append((c.x-d.x*offset+7.4/math.sqrt(3)*math.cos(t),c.y-d.y*offset+7.4/math.sqrt(3)*math.sin(t)))
 poly=hull2(pts);N=len(poly);vs=[(x,y,z)for z in [98.6,102.2]for x,y in poly];fs=[tuple(reversed(range(N))),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N)for i in range(N)];me=bpy.data.meshes.new('nut-entry');me.from_pydata(vs,[],fs);me.update();o=bpy.data.objects.new('nut-entry',me);bpy.context.collection.objects.link(o);clean(o);boolean(head,o,'DIFFERENCE')
# Split before making the blind alignment pockets, so each pocket opens at its mating face.
for _ in range(3):clean(head)
report={'head-reference':export(head,'head')}
bpy.data.objects.remove(head,do_unlink=True);bpy.ops.wm.stl_import(filepath=str(out/'head.stl'));head=bpy.context.object;clean(head)
def cube(lo,hi):
 bpy.ops.mesh.primitive_cube_add(size=1,location=[(lo[i]+hi[i])/2 for i in range(3)]);o=bpy.context.object;o.dimensions=[hi[i]-lo[i]for i in range(3)];bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
upper=head.copy();upper.data=head.data.copy();bpy.context.collection.objects.link(upper);upper.name='head-upper';boolean(upper,cube([-500,-500,-500],[500,500,Z]),'DIFFERENCE');lower=head.copy();lower.data=head.data.copy();bpy.context.collection.objects.link(lower);lower.name='head-lower';boolean(lower,cube([-500,-500,Z],[500,500,500]),'DIFFERENCE')
for p in pins:
 c=Vector(p['center'])
 for obj,zz in [(lower,105.15),(upper,108.25)]:
  bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=2.2,depth=3.3,location=(c.x,c.y,zz));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(obj,o,'DIFFERENCE')
axis=(Vector(pins[1]['center'])-Vector(pins[0]['center'])).normalized();c=Vector(pins[1]['center'])
for d in [-.4,.4]:
 cc=c+axis*d;bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=2.2,depth=3.3,location=(cc.x,cc.y,108.25));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(upper,o,'DIFFERENCE')
for n,o in [('head-upper',upper),('head-lower',lower)]:
 rep=export(o,n);rep['worldBounds']=[[min(v.co[k]for v in o.data.vertices),max(v.co[k]for v in o.data.vertices)]for k in range(3)];report[n]=rep
report.update({'splitZMM':Z,'flanges':selected,'pins':pins,'hardware':{'screws':'4x M4x20','washers':'4x M4 OD9 t0.5','nuts':'4x M4 AF7 t3.2','nutShoulderZMM':102.2,'nutZMM':[98.9,102.1],'screwUnderHeadZMM':117.2,'screwTipZMM':97.2,'boltInsertion':'from upper shell; nuts slide inward-to-outward into lower shell channels','nutPreRetention':'small epoxy spot on nut side only, no adhesive on bearing face or thread'},'headSource':'v12 upper and lower rejoined; new seam Z106.7; existing optical fittings preserved','physicalTests':False})
(out/'head-split-check.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
