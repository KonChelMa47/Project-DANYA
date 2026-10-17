exec(open('work/face-v9/build-candidates.py').read().split('rows=[]')[0])
def loadFile(p,name):
 bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;o.name=name;clean(o);return o
base=loadFile(r/'archive-before-integrated-magnets/assembly/head.stl','original-head')
upper=loadFile(r/'assembly/head-upper.stl','head-upper');lower=loadFile(r/'assembly/head-lower.stl','head-lower');face=loadFile(r/'face-v8/screen-after.stl','face')
for i in range(4):
 pure=loadFile(r/'magnets-v2'/f'integrated-{i}-head-addition.stl','old-holder');boolean(pure,base,'DIFFERENCE',False)
 for h in [upper,lower]:boolean(h,pure,'DIFFERENCE',False)
 print('removed old head cup',i,flush=True)
for row in json.load(open(out/'positions.json')):
 i=row['id'];relief=loadFile(out/f'head-relief-{i}.stl','head-relief');boolean(upper if row['top']else lower,relief,'DIFFERENCE');holder=loadFile(out/f'head-holder-{i}.stl',f'new-head-holder-{i}');boolean(upper if row['top']else lower,holder,'UNION')
 arm=loadFile(out/f'face-holder-{i}.stl',f'new-face-holder-{i}');boolean(face,arm,'UNION');print('joined pair',i,flush=True)
reports={}
for name,obj in [('face',face),('head-upper',upper),('head-lower',lower)]:
 rep=export(obj,name);reports[name]=rep;print(name,rep,flush=True);assert rep['components']==1 and rep['nonManifoldEdges']==0,rep
(out/'geometry-check.json').write_text(json.dumps(reports,indent=2))
