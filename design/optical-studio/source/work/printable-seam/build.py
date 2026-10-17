exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'printable-seam';out.mkdir(exist_ok=True);bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def op(a,b,mode):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def meshband(inside,outside,name):
 S,N=inside.shape[:2];vv=np.concatenate([inside.reshape(-1,3),outside.reshape(-1,3)]);L=S*N;ff=[]
 for layer in [0,1]:
  for j in range(S-1):
   for i in range(N):a=layer*L+j*N+i;b=layer*L+j*N+(i+1)%N;ff.append((a,b,b+N,a+N)if layer else(a,a+N,b+N,b))
 for j in [0,S-1]:
  for i in range(N):a=j*N+i;b=j*N+(i+1)%N;ff.append((a,b,b+L,a+L))
 me=bpy.data.meshes.new(name);me.from_pydata(vv.tolist(),[],ff);me.update();o=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(o);clean(o);return o
p=np.load(src/'face-v12/perimeter-paths.npz');paths=p['paths'];dirs=p['dirs'];S,N=paths.shape[:2];inside=paths-dirs[None,:,:]*np.linspace(.25,.85,S)[:,None,None]
tangent=np.roll(inside,-1,axis=1)-np.roll(inside,1,axis=1);axial=np.gradient(inside,axis=0);normal=np.cross(tangent,axial);normal/=np.maximum(np.linalg.norm(normal,axis=2)[:,:,None],1e-9);factor=np.abs(np.sum(normal*dirs[None,:,:],axis=2));radial=1.4/np.maximum(factor,.45);outside=inside+dirs[None,:,:]*radial[:,:,None]
# Smooth the visible rim without reducing the requested wall thickness: max-filter then a smaller averaging kernel.
center=np.array([0,0,160]);rho=radial.copy();expanded=np.max(np.stack([np.roll(rho,k,axis=1)for k in range(-16,17)]),axis=0);smooth=sum(np.roll(expanded,k,axis=1)for k in range(-8,9))/17
outside+=dirs[None,:,:]*(smooth-rho)[:,:,None]
np.savez(out/'band-samples.npz',carrierInside=inside,carrierOutside=outside);band=meshband(inside,outside,'carrier-band-1.4mm');export(band,'carrier-band')
bpy.ops.wm.stl_import(filepath=str(src/'assembly/magnet-carrier.stl'));carrier=bpy.context.object;op(carrier,band,'UNION');print('CARRIER',export(carrier,'magnet-carrier'),flush=True);bpy.data.objects.remove(carrier,do_unlink=True)
# Replace the thin overlapping rear-head lip by a butt seam, with 0.30mm axial clearance.
rear=paths[-1].copy();rear[:,0]-=.30;maskIn=np.array([rear-dirs*2.0,rear-dirs*2.0+np.array([70,0,0])]);maskOut=np.array([rear+dirs*8,rear+dirs*8+np.array([70,0,0])]);tool=meshband(maskIn,maskOut,'remove-thin-overlap');export(tool,'thin-lip-removal-tool');bpy.data.objects.remove(tool,do_unlink=True)
# Reinforce externally; preserve the original inner light envelope. Blend to original skull over8mm.
bpy.ops.wm.stl_import(filepath=str(src/'archive-before-printable-mounts-v5/assembly/head.stl'));ref=bpy.context.object;tree=BVHTree.FromPolygons([v.co for v in ref.data.vertices],[f.vertices[:]for f in ref.data.polygons]);hin=[];hout=[];miss=0
for j,t in enumerate(np.linspace(0,1,9)):
 rowIn=[];rowOut=[]
 for i in range(N):
  x=rear[i,0]-8*t;origin=Vector((float(x),0,160));d=Vector(dirs[i]);start=origin.copy();hits=[]
  for _ in range(12):
   h=tree.ray_cast(start,d,250)[0]
   if h is None:break
   hits.append(np.array(h));start=h+d*.005
  if hits:outer=hits[-1]
  else:outer=rear[i].copy();outer[0]=x;miss+=1
  want=np.dot(outside[-1,i]-outer,dirs[i]);lift=max(.5,min(1.8,want))*(1-t)**1.2;rowIn.append(outer-dirs[i]*.45);rowOut.append(outer+dirs[i]*lift)
 hin.append(rowIn);hout.append(rowOut)
bpy.data.objects.remove(ref,do_unlink=True);np.savez(out/'head-band-samples.npz',inside=np.array(hin),outside=np.array(hout));reinforce=meshband(np.array(hin),np.array(hout),'head-band');export(reinforce,'head-band');bpy.data.objects.remove(reinforce,do_unlink=True)
def cube(lo,hi):
 bpy.ops.mesh.primitive_cube_add(size=1,location=[(a+b)/2 for a,b in zip(lo,hi)]);o=bpy.context.object;o.dimensions=[b-a for a,b in zip(lo,hi)];bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
reports={}
for name in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(src/'head-split-quarter'/(name+'.stl')));head=bpy.context.object
 bpy.ops.wm.stl_import(filepath=str(out/'thin-lip-removal-tool.stl'));op(head,bpy.context.object,'DIFFERENCE')
 bpy.ops.wm.stl_import(filepath=str(out/'head-band.stl'));band=bpy.context.object;clip=cube([-500,-500,106.7 if name=='head-upper' else -500],[500,500,500 if name=='head-upper' else 106.7]);op(band,clip,'INTERSECT');op(head,band,'UNION');reports[name]=export(head,name);print(name,reports[name],flush=True);bpy.data.objects.remove(head,do_unlink=True)
(out/'build-report.json').write_text(json.dumps({'heads':reports,'carrierNormalTargetMM':1.4,'seamAxialClearanceMM':.3,'headBandBlendLengthMM':8,'headRayMisses':miss,'maxCarrierOutwardAdditionMM':float(np.max(radial-np.linspace(.25,.85,S)[:,None]))},indent=2))
