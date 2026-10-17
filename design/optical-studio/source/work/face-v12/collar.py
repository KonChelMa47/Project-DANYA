exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
from mathutils import Vector
from mathutils.bvhtree import BVHTree
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
d=np.load(r/'face-v8/screen-data.npz');v=d['after'];f=d['outerFaces'];edges=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);u,c=np.unique(edges,axis=0,return_counts=True);boundary=u[c==1];adj={}
for a,b in boundary:adj.setdefault(int(a),[]).append(int(b));adj.setdefault(int(b),[]).append(int(a))
assert all(len(a)==2 for a in adj.values());ids=[next(iter(adj))];prev=None
while True:
 nxt=next(n for n in adj[ids[-1]]if n!=prev)
 if nxt==ids[0]:break
 prev=ids[-1];ids.append(nxt)
pts=v[ids];ds=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1);cum=np.r_[0,np.cumsum(ds)];N=720;curve=np.array([np.interp(np.arange(N)*cum[-1]/N,cum,np.r_[pts[:,k],pts[0,k]])for k in range(3)]).T
head=load(r/'archive-before-integrated-magnets/assembly/head.stl');tree=BVHTree.FromPolygons([p.co for p in head.data.vertices],[p.vertices[:]for p in head.data.polygons]);bpy.data.objects.remove(head,do_unlink=True)
front=curve.copy();rear=[];dirs=[];miss=[]
for i,p in enumerate(front):
 direction=Vector((0,float(p[1]),float(p[2]-160))).normalized();dirs.append(np.array(direction));p+=np.array(direction)*.35;p[0]-=.15
 x=p[0]-(8+4*np.clip((p[2]-230)/15,0,1));start=Vector((float(x),0,160));hits=[]
 for _ in range(12):
  h=tree.ray_cast(start,direction,250)[0]
  if h is None:break
  hits.append(np.array(h));start=h+direction*.003
 if not hits:miss.append(i);rear.append(p-np.array([8,0,0]))
 else:rear.append(hits[-1]+np.array(direction)*.02)
assert not miss,miss
rear=np.array(rear);dirs=np.array(dirs);S=9;vv=[]
beam=load(r/'face-v10/inner-cavity.stl');bt=BVHTree.FromPolygons([p.co for p in beam.data.vertices],[p.vertices[:]for p in beam.data.polygons]);bpy.data.objects.remove(beam,do_unlink=True)
paths=[];lifts=[]
for j in range(S):
 t=j/(S-1);p=front*(1-t)+rear*t+dirs*(.10*np.sin(np.pi*t));thickness=.25+.60*t
 for i,q in enumerate(p):
  origin=Vector((float(q[0]),0,160));direction=Vector(dirs[i]);hits=[];start=origin.copy()
  for _ in range(8):
   hit=bt.ray_cast(start,direction,250)[0]
   if hit is None:break
   hits.append((hit-origin).length);start=hit+direction*.003
  if hits:
   radius=np.linalg.norm(q[1:]-np.array([0,160]));lift=max(0,max(hits)+thickness+.12-radius);q+=dirs[i]*lift;lifts.append(lift)
 paths.append(p)
for inner in [False,True]:
 for j,p in enumerate(paths):vv.extend((p-dirs*((.25+.6*j/(S-1))if inner else 0)).tolist())
print('optical envelope outward allowance max',max(lifts),flush=True)
ff=[];L=N*S
for layer in [0,1]:
 for j in range(S-1):
  for i in range(N):a=layer*L+j*N+i;b=layer*L+j*N+(i+1)%N;ff.append((a,b,b+N,a+N)if layer==0 else(a,a+N,b+N,b))
for j in [0,S-1]:
 for i in range(N):a=j*N+i;b=j*N+(i+1)%N;ff.append((a,a+L,b+L,b))
m=bpy.data.meshes.new('smooth anatomical perimeter');m.from_pydata(vv,[],ff);m.update();cover=bpy.data.objects.new('perimeter',m);bpy.context.collection.objects.link(cover);clean(cover);print('cover',export(cover,'smooth-perimeter'),flush=True)
# Closed outer skin envelope clips old support protrusions to the anatomical rim.
np.savez(out/'perimeter-paths.npz',paths=np.array(paths),dirs=dirs)
loops=[paths[0]+np.array([200,0,0])]+paths+[paths[-1]-np.array([200,0,0])]
evs=np.concatenate(loops).tolist();efs=[]
for j in range(len(loops)-1):
 for i in range(N):a=j*N+i;b=j*N+(i+1)%N;efs.append((a,b,b+N,a+N))
for j in [0,len(loops)-1]:
 idx=len(evs);evs.append(np.mean(loops[j],axis=0).tolist())
 for i in range(N):efs.append((idx,j*N+i,j*N+(i+1)%N))
mm=bpy.data.meshes.new('outer skin envelope');mm.from_pydata(evs,[],efs);mm.update();env=bpy.data.objects.new('outer skin envelope',mm);bpy.context.collection.objects.link(env);clean(env);export(env,'skin-envelope')
o=load(out/'magnet-carrier.stl');cut(o,box((21.75,0,275),(40,60,50)));union(o,cover);intersect(o,env)
# Keep the optical face aperture open while retaining the smooth outer skin.
# Collar inner surface stays outside the sampled optical envelope; no perforating trim.
for y in [-18.054,18.054]:
 cut(o,cylinder_at((42.3,y,255.0),1.65,1.4))
 cut(o,cylinder_at((36.75,y,255.0),1.7,10))
print('carrier',export(o,'magnet-carrier-covered'),flush=True)
# Internal receiving rebate: exterior is restored by the flush removable perimeter.
for n in ['head-upper','head-lower']:
 h=load(out/(n+'.stl'))
 verts=[]
 for layer in [0,1]:
  for j in range(S):
   t=j/(S-1);p=paths[j].copy()-dirs*((.25+.6*t)if layer else 0);p+=dirs*(-.25 if layer else .25);p[:,0]+=.25-.5*t;verts.extend(p.tolist())
 m=bpy.data.meshes.new('clearance rebate');m.from_pydata(verts,[],ff);m.update();tool=bpy.data.objects.new('clearance rebate',m);bpy.context.collection.objects.link(tool);clean(tool);cut(h,tool)
 if n=='head-upper':
  for y in [-18.054,18.054]:cut(h,cylinder_at((51.55,y,255.0),2.5,20))
 print(n,export(h,n+'-covered'),flush=True)
