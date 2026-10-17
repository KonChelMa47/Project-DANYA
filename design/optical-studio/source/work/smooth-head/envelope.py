exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'smooth-head-v4';fit=np.load(out/'fit.npz');C=fit['C'];scale=fit['scale'];coef=fit['coef'];powers=fit['powers']
def target(v):
 u=(v-C)/scale;rr=np.linalg.norm(u,axis=1);u/=rr[:,None];A=np.stack([u[:,0]**i*u[:,1]**j*u[:,2]**k for i,j,k in powers],1);return C+u*(A@coef)[:,None]*scale
D=np.load(src/'printable-seam/band-samples.npz');outer=D['carrierOutside'][-1].copy();inner=D['carrierInside'][-1].copy();outer[:,0]-=.3;inner[:,0]=outer[:,0];N=len(outer);S=180
# Parameterize a single continuous disk from the rear pole to the face perimeter.
surfaces={}
for name,edge,thick in [('outer-envelope',outer,0),('inner-envelope',inner,4.3)]:
 dirs=(outer-C);dirs/=np.linalg.norm(dirs,axis=1)[:,None];angles=np.arccos(-dirs[:,0]);yz=dirs[:,1:];yz/=np.linalg.norm(yz,axis=1)[:,None];vs=[]
 for j in range(1,S+1):
  t=j/S;a=angles*t;ray=np.column_stack([-np.cos(a),np.sin(a)[:,None]*yz]);p=target(C+ray*100)
  if thick:
   # Offset along the fitted surface normal using two finite differences.
   eps=.002;u=ray;v1=np.cross(u,np.array([0,0,1.]));v1/=np.maximum(np.linalg.norm(v1,axis=1)[:,None],1e-9);v2=np.cross(u,v1);t1=target(C+(u+eps*v1)*100)-target(C+(u-eps*v1)*100);t2=target(C+(u+eps*v2)*100)-target(C+(u-eps*v2)*100);nn=np.cross(t1,t2);nn/=np.linalg.norm(nn,axis=1)[:,None];nn*=np.sign(np.sum(nn*u,axis=1))[:,None];p-=nn*thick
  if j==117:pstart=p.copy();dstart=(p-pbefore)*S*.35
  if j>117:
   h=(t-.65)/.35;enddir=D['carrierOutside' if not thick else 'carrierInside'][-2]-D['carrierOutside' if not thick else 'carrierInside'][-1];enddir/=np.linalg.norm(enddir,axis=1)[:,None];dend=enddir*np.linalg.norm(edge-pstart,axis=1)[:,None]
   p=(2*h**3-3*h**2+1)*pstart+(h**3-2*h**2+h)*dstart+(-2*h**3+3*h**2)*edge+(h**3-h**2)*dend
  pbefore=p.copy()
  if j==S:p=edge.copy()
  vs.extend(p.tolist())
 pole=target(np.array([C+[-100,0,0]]))[0];pole[0]+=thick;vs.append(pole.tolist());poleid=len(vs)-1;center=np.mean(edge,axis=0);vs.append(center.tolist());cid=len(vs)-1;ff=[]
 for i in range(N):ff.append((poleid,i,(i+1)%N))
 for j in range(S-1):
  for i in range(N):a=j*N+i;b=j*N+(i+1)%N;c=b+N;d=a+N;ff.extend([(a,d,c),(a,c,b)])
 for i in range(N):ff.append(((S-1)*N+i,cid,(S-1)*N+(i+1)%N))
 surfaces[name]=(vs,ff)
 if name=='outer-envelope':
  ff=ff[:-N];extstart=len(vs);vs.extend([[100,p[1],p[2]]for p in edge]);vs[cid]=[100,*center[1:]]
  for i in range(N):a=(S-1)*N+i;b=(S-1)*N+(i+1)%N;c=extstart+(i+1)%N;d=extstart+i;ff.extend([(a,b,c),(a,c,d),(d,c,cid)])
 me=bpy.data.meshes.new(name);me.from_pydata(vs,[],ff);me.update();o=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(o);print(name,export(o,name),flush=True)
# Join only the two smooth surfaces and the perimeter rim; no internal fan-shaped membrane.
va,fa=surfaces['outer-envelope'];_,fb=surfaces['inner-envelope'];va=va[:S*N+1];grid=np.array(va[:S*N]).reshape(S,N,3);dt=np.gradient(grid,axis=0);da=np.roll(grid,-1,axis=1)-np.roll(grid,1,axis=1);norm=np.cross(dt,da);norm/=np.maximum(np.linalg.norm(norm,axis=2)[:,:,None],1e-9);norm*=np.sign(np.sum(norm*(grid-C),axis=2))[:,:,None];ig=grid-norm*4.3;dist=np.linalg.norm(grid-grid[-1],axis=2);w=np.clip(1-dist/6,0,1);w=w*w*(3-2*w);ig+=(inner-ig[-1])[None,:,:]*w[:,:,None];vb=ig.reshape(-1,3).tolist()+[(np.array(va[-1])+np.array([4.3,0,0])).tolist()];L=len(va);ff=fa[:-N]+[tuple(i+L for i in reversed(f))for f in fb[:-N]]
for i in range(N):a=(S-1)*N+i;b=(S-1)*N+(i+1)%N;ff.extend([(a,b,b+L),(a,b+L,a+L)])
me=bpy.data.meshes.new('smooth shell');me.from_pydata(va+vb,[],ff);me.update();o=bpy.data.objects.new('smooth shell',me);bpy.context.collection.objects.link(o);print('SHELL',export(o,'smooth-shell'),flush=True)
