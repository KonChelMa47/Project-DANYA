exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
from mathutils.kdtree import KDTree
out=src/'smooth-head-v4';fit=np.load(out/'fit.npz');C=fit['C'];scale=fit['scale'];coef=fit['coef'];powers=fit['powers']
def target(v):
 u=(v-C)/scale;rr=np.linalg.norm(u,axis=1);u/=rr[:,None];A=np.stack([u[:,0]**i*u[:,1]**j*u[:,2]**k for i,j,k in powers],1);return C+u*(A@coef)[:,None]*scale
objs=[loadpath(src/'thick-head-v3'/(name+'.stl'))for name in ['head-upper','head-lower']];av=[];af=[]
for o in objs:
 k=len(av);av.extend(v.co.copy()for v in o.data.vertices);af.extend(tuple(k+i for i in p.vertices)for p in o.data.polygons)
tree=BVHTree.FromPolygons(av,af);carrier=np.load(src/'printable-seam/band-samples.npz')['carrierOutside'][-1];kd=KDTree(len(carrier))
for i,p in enumerate(carrier):kd.insert(Vector(p),i)
kd.balance();report={}
for o,name in zip(objs,['head-upper','head-lower']):
 v=np.array([p.co[:]for p in o.data.vertices]);q=target(v);moves=[];nmove=0
 for i,p in enumerate(v):
  d=Vector(p-C).normalized();h=tree.ray_cast(Vector(p)+d*300,-d,301)[0]
  if h is None or (h-Vector(p)).length>.06:continue
  _,idx,dist=kd.find(Vector(p));edge=carrier[idx];frontdist=edge[0]-.3-p[0]
  if frontdist<-.1 or p[2]<50.649:continue
  if np.linalg.norm(q[i]-p)>10:continue
  # Blend fitted skull into the existing carrier outside at the face seam.
  t=np.clip(frontdist/25,0,1);w=t*t*(3-2*t);delta=q[i]-p
  if t<1:
   eyz=edge[1:]-C[1:];rr=np.linalg.norm(eyz);rad=(p[1:]-C[1:]);rad/=max(np.linalg.norm(rad),1e-9)
   # Match the carrier envelope, keeping seam X unchanged.
   dd=np.array([0,*((rr-np.linalg.norm(p[1:]-C[1:]))*rad)])
   delta=delta*w+dd*(1-w)
  if abs(p[2]-165)<.0001:delta[2]=0
  if p[2]<50.651:delta[2]=0
  # Do not move the functional magnet cup tips or their bores.
  if p[0]>edge[0]-2:continue
  o.data.vertices[i].co=Vector(p+delta);moves.append(np.linalg.norm(delta));nmove+=1
 print(name,'moved',nmove,'max',max(moves),flush=True);report[name]={'movedVertices':nmove,'maxMM':max(moves)};print(export(o,name),flush=True)
(out/'smoothing.json').write_text(json.dumps(report,indent=2))
