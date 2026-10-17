exec(open('work/face-v9/build-candidates.py').read().split('rows=[]')[0])
def loadFile(p,name):
 bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;o.name=name;clean(o);return o
base=loadFile(r/'archive-before-integrated-magnets/assembly/head.stl','original-head')
upper=loadFile(r/'assembly/head-upper.stl','head-upper');lower=loadFile(r/'assembly/head-lower.stl','head-lower');face=loadFile(out/'screen-after.stl','face')
for i in range(4):
 pure=loadFile(r/'magnets-v2'/f'integrated-{i}-head-addition.stl','old-holder');boolean(pure,base,'DIFFERENCE',False)
 for h in [upper,lower]:boolean(h,pure,'DIFFERENCE',False)
 print('removed old head cup',i,flush=True)
# Transfer a 6mm band of the original rear shell to the face: continuous structural rim,
# assembled outer surface unchanged. The actual exterior division line moves rearward.
vv=[];N=81
for layer in [0,1]:
 for i in range(N):
  yy=-110+220*i/(N-1)
  for j in range(N):
   zz=30+270*j/(N-1);xx=seam(yy,zz)-6 if layer==0 else 500
   vv.append((xx,yy,zz))
front=triangles(grid(vv,N));export(front,'collar-partition-tool')
for h in [upper,lower]:
 collar=h.copy();collar.data=h.data.copy();bpy.context.collection.objects.link(collar)
 boolean(collar,front,'INTERSECT',False);boolean(h,front,'DIFFERENCE',False);bpy.context.view_layer.objects.active=face;m=face.modifiers.new('join rim','BOOLEAN');m.operation='UNION';m.solver='EXACT';m.object=collar;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(collar,do_unlink=True)
 print('transferred collar',h.name,flush=True)
for row in json.load(open(out/'positions.json')):
 i=row['id'];relief=loadFile(out/f'head-relief-{i}.stl','head-relief');boolean(upper if row['top']else lower,relief,'DIFFERENCE');holder=loadFile(out/f'head-holder-{i}.stl',f'new-head-holder-{i}');boolean(upper if row['top']else lower,holder,'UNION')
 arm=loadFile(out/f'face-holder-{i}.stl',f'new-face-holder-{i}');boolean(face,arm,'UNION');print('joined pair',i,flush=True)
reports={}
for name,obj in [('head-upper',upper),('head-lower',lower),('face',face)]:
 rep=export(obj,name);reports[name]=rep;print(name,rep,flush=True)
(out/'geometry-check.json').write_text(json.dumps(reports,indent=2))
