exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
def direct(a,b,mode):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def box(c,d):
 bpy.ops.mesh.primitive_cube_add(size=1,location=c);o=bpy.context.object;o.dimensions=d;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
ref=loadpath(src/'solid-seam-v2/head-lower.stl');tree=BVHTree.FromPolygons([v.co for v in ref.data.vertices],[p.vertices[:]for p in ref.data.polygons]);N=360;zs=np.linspace(50.65,130,81);S=len(zs);vi=np.zeros((S,N,3));vo=vi.copy();valid=np.zeros((S,N),bool)
for j,z in enumerate(zs):
 for i in range(N):
  a=i*math.tau/N;d=Vector((math.cos(a),math.sin(a),0));p=Vector((-22,0,float(z)+.001));start=p.copy();hits=[];normal=None
  for _ in range(20):
   h,nn,_,_=tree.ray_cast(start,d,220)
   if h is None:break
   hits.append(h);normal=nn;start=h+d*.003
  if len(hits)<2 or len(hits)%2:continue
  outer=hits[-1];inner=hits[-2];factor=max(.5,abs(normal.dot(d)));t=(outer-inner).length;off=max(.05,min(5,(4.2-t*factor)/factor));valid[j,i]=True;vi[j,i]=outer-d*.35;vo[j,i]=outer+d*off;vi[j,i,2]=z;vo[j,i,2]=z
bpy.data.objects.remove(ref,do_unlink=True);L=S*N;vv=np.concatenate([vi.reshape(-1,3),vo.reshape(-1,3)]);ff=[];edges={}
for j in range(S-1):
 for i in range(N):
  ids=[j*N+i,j*N+(i+1)%N,(j+1)*N+(i+1)%N,(j+1)*N+i]
  if not all(valid.flat[k]for k in ids):continue
  ff.extend([tuple(reversed(ids)),tuple(k+L for k in ids)])
  for a,b in zip(ids,ids[1:]+ids[:1]):edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
for es in edges.values():
 if len(es)==1:a,b=es[0];ff.append((a,b,b+L,a+L))
me=bpy.data.meshes.new('reinforced neck');me.from_pydata(vv.tolist(),[],ff);me.update();band=bpy.data.objects.new('reinforced neck',me);bpy.context.collection.objects.link(band);clean(band)
direct(band,loadpath(src/'printable-seam/thin-lip-removal-tool.stl'),'DIFFERENCE');direct(band,box((150,0,80),(300,300,200)),'DIFFERENCE');print('NECK',export(band,'neck-band'),flush=True)

head=loadpath(out/'head-lower-final.stl');direct(head,band,'UNION');print(export(head,'head-lower-neckfixed'))
