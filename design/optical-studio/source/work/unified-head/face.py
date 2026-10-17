exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
from mathutils.kdtree import KDTree
out=src/'unified-head-v5';d=np.load(src/'face-v8/screen-data.npz');v=d['after'].copy();f=d['faces'];outerIDs=np.unique(d['outerFaces']);curves=np.load(out/'band-samples.npz');old=curves['faceBoundaryOriginal'];new=curves['faceBoundary'];N=len(old);kd=KDTree(N)
for i,p in enumerate(old):kd.insert(Vector(p),i)
kd.balance();moved=[]
for i in outerIDs:
 p=v[i];_,idx,_=kd.find(Vector(p));best=None
 for j in [(idx-1)%N,idx]:
  a=old[j];b=old[(j+1)%N];delta=b-a;t=np.clip(np.dot(p-a,delta)/np.dot(delta,delta),0,1);near=a+delta*t;dist=np.linalg.norm(p-near)
  if best is None or dist<best[0]:best=(dist,j,t)
 dist,j,t=best
 if dist>=10:continue
 u=dist/10;w=1-u*u*(3-2*u);change=((new[j]-old[j])*(1-t)+(new[(j+1)%N]-old[(j+1)%N])*t)*w;v[i]+=change;moved.append(np.linalg.norm(change))
me=bpy.data.meshes.new('face smooth perimeter');me.from_pydata(v.tolist(),[],f.tolist());me.update();o=bpy.data.objects.new('face',me);bpy.context.collection.objects.link(o);print(export(o,'face'));(out/'face-perimeter-change.json').write_text(json.dumps({'outerVerticesModified':len(moved),'blendDistanceFromEdgeMM':10,'maxDisplacementMM':max(moved),'medianDisplacementMM':float(np.median(moved)),'innerProjectionSurface':'unchanged screen-data v8 after','features':'vertices more than10mm from original boundary unchanged'},indent=2))
