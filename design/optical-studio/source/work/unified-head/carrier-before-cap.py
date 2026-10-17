exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'unified-head-v5';d=np.load(out/'band-samples.npz');inside=d['carrierInside'];outside=d['carrierOutside'];S,N=outside.shape[:2]
def direct(a,b,mode):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def mesh(v,f,name):
 m=bpy.data.meshes.new(name);m.from_pydata(v,[],f);m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);clean(o);return o
v=np.concatenate([outside.reshape(-1,3),inside.reshape(-1,3)]);L=S*N;f=[]
for j in range(S-1):
 for i in range(N):a=j*N+i;b=j*N+(i+1)%N;f.extend([(a,b,b+N),(a,b+N,a+N),(a+L,b+N+L,b+L),(a+L,a+N+L,b+N+L)])
for j in [0,S-1]:
 for i in range(N):a=j*N+i;b=j*N+(i+1)%N;f.extend([(a,a+L,b+L),(a,b+L,b)])
band=mesh(v.tolist(),f,'newcarrier');print('band',export(band,'carrier-skin'),flush=True)
# Clip old supports beneath the new skin so no old exterior patches remain visible.
rad=outside-np.array([0,0,160]);rad[:,:,0]=0;rad/=np.linalg.norm(rad,axis=2)[:,:,None];profile=outside-rad*.45;loops=np.concatenate([profile[:1]+[100,0,0],profile,profile[-1:]-[100,0,0]]);v=loops.reshape(-1,3).tolist();f=[]
for j in range(len(loops)-1):
 for i in range(N):a=j*N+i;b=j*N+(i+1)%N;f.extend([(a,b,b+N),(a,b+N,a+N)])
for j in [0,len(loops)-1]:
 k=len(v);v.append(loops[j].mean(0).tolist())
 for i in range(N):f.append((k,j*N+i,j*N+(i+1)%N))
env=mesh(v,f,'carrier inset envelope');print(export(env,'carrier-inset-envelope'),flush=True);old=loadpath(src/'solid-seam-v2/magnet-carrier.stl');direct(old,env,'INTERSECT');direct(old,band,'UNION');print(export(old,'magnet-carrier'),flush=True)
