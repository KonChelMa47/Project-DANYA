exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
from mathutils import Vector
from mathutils.bvhtree import BVHTree
out=src/'solid-seam-v2';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def op(a,b,mode):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def box(c,d):
 bpy.ops.mesh.primitive_cube_add(size=1,location=c);o=bpy.context.object;o.dimensions=d;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
p=np.load(src/'face-v12/perimeter-paths.npz');paths=p['paths'];dirs=p['dirs'];rear=paths[-1].copy();rear[:,0]-=.3;target=np.load(src/'printable-seam/band-samples.npz')['carrierOutside'][-1];N=len(rear);S=13;reports={}
for name in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(src/'head-split-quarter'/(name+'.stl')));head=bpy.context.object;tree=BVHTree.FromPolygons([v.co for v in head.data.vertices],[f.vertices[:]for f in head.data.polygons]);inside=np.zeros((S,N,3));outside=inside.copy();valid=np.zeros((S,N),bool)
 for j,t in enumerate(np.linspace(0,1,S)):
  for i in range(N):
   x=rear[i,0]-8*t;origin=Vector((float(x),0,160));d=Vector(dirs[i]);start=origin.copy();hits=[]
   for _ in range(20):
    h=tree.ray_cast(start,d,250)[0]
    if h is None:break
    hits.append(np.array(h));start=h+d*.002
   if len(hits)<2 or len(hits)%2:continue
   outer=hits[-1];inner=hits[-2]
   # Never synthesize material where there is no actual head wall, or over a hardware cavity.
   if np.linalg.norm(outer-rear[i]+np.array([8*t,0,0]))>5:continue
   if (name=='head-upper'and outer[2]<106.72)or(name=='head-lower'and outer[2]>106.68):continue
   valid[j,i]=True;inside[j,i]=outer-dirs[i]*min(.7,max(.2,np.linalg.norm(outer-inner)*.75));want=np.dot(target[i]-outer,dirs[i]);lift=max(.6,min(1.8,want))*(1-t)**1.2+.02;outside[j,i]=outer+dirs[i]*lift
 cells=[]
 for j in range(S-1):
  for i in range(N):
   ids=[j*N+i,j*N+(i+1)%N,(j+1)*N+(i+1)%N,(j+1)*N+i]
   if all(valid.flat[k]for k in ids):cells.append(ids)
 # Closed bands assembled only over real wall cells. Their boundary caps overlap the existing shell.
 vv=np.concatenate([inside.reshape(-1,3),outside.reshape(-1,3)]);L=S*N;ff=[];edges={}
 for ids in cells:
  ff.extend([tuple(reversed(ids)),tuple(k+L for k in ids)])
  for a,b in zip(ids,ids[1:]+ids[:1]):key=tuple(sorted((a,b)));edges.setdefault(key,[]).append((a,b))
 for es in edges.values():
  if len(es)==1:a,b=es[0];ff.append((a,b,b+L,a+L))
 me=bpy.data.meshes.new('real wall reinforcement');me.from_pydata(vv.tolist(),[],ff);me.update();band=bpy.data.objects.new('real wall reinforcement',me);bpy.context.collection.objects.link(band);clean(band)
 # Trim the prior wafer-like overlap, but protect all magnet seats and their existing roots.
 bpy.ops.wm.stl_import(filepath=str(src/'printable-seam/thin-lip-removal-tool.stl'));tool=bpy.context.object
 for y,z in [(-18.054,255),(18.054,255),(-42,62),(42,62)]:op(tool,box((38,y,z),(25,12,14)),'DIFFERENCE')
 op(head,tool,'DIFFERENCE');op(head,band,'UNION');reports[name]=export(head,name);reports[name]['validCells']=len(cells);print(name,reports[name],flush=True);bpy.data.objects.remove(head,do_unlink=True)
(out/'build-report.json').write_text(json.dumps(reports,indent=2))
