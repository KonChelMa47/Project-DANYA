exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
d=np.load(out/'perimeter-paths.npz');paths=d['paths'];dirs=d['dirs'];S,N=paths.shape[:2];vv=[];ff=[];L=S*N
for layer in [0,1]:
 for j in range(S):
  t=j/(S-1);p=paths[j].copy()-dirs*((.25+.6*t)if layer else 0);p+=dirs*(-.3 if layer else .3);p[:,0]+=.3-.6*t;vv.extend(p.tolist())
for layer in [0,1]:
 for j in range(S-1):
  for i in range(N):a=layer*L+j*N+i;b=layer*L+j*N+(i+1)%N;ff.append((a,b,b+N,a+N)if layer==0 else(a,a+N,b+N,b))
for j in [0,S-1]:
 for i in range(N):a=j*N+i;b=j*N+(i+1)%N;ff.append((a,a+L,b+L,b))
m=bpy.data.meshes.new('final inner rebate');m.from_pydata(vv,[],ff);m.update();tool=bpy.data.objects.new('final inner rebate',m);bpy.context.collection.objects.link(tool);clean(tool)
o=load(out/'head-upper-covered.stl');clean(o);cut(o,tool);print(export(o,'head-upper-covered'),flush=True)
