exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'thick-head-v3';bpy.ops.wm.read_factory_settings(use_empty=True)
def op(a,b,mode,remove=True):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
def loadpath(p):
 bpy.ops.wm.stl_import(filepath=str(p));return bpy.context.object
head=loadpath(out/'jointless.stl')
me=head.data;verts=np.array([v.co[:]for v in me.vertices]);faces=[p.vertices[:]for p in me.polygons];tree=BVHTree.FromPolygons([Vector(v)for v in verts],faces);C=Vector((-22,0,160));keep=[]
for p in me.polygons:
 cen=p.center;d=cen-C
 if d.length<1:continue
 d.normalize()
 if p.normal.dot(d)<.25:continue
 h=tree.ray_cast(cen+d*300,-d,301)[0]
 if h is not None and (h-cen).length<.025:keep.append(list(p.vertices))
print('OUTER FACES',len(keep),flush=True)
used=sorted({i for f in keep for i in f});ix={v:i for i,v in enumerate(used)};fv=np.array([[ix[i]for i in f]for f in keep]);v=verts[used];norm=np.zeros_like(v);neighbors=[set()for _ in used]
for f in fv:
 a,b,c=v[f[:3]];nn=np.cross(b-a,c-a)
 for i in f:norm[i]+=nn;neighbors[i].update(j for j in f if j!=i)
norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-12)
for _ in range(8):
 norm=np.array([.5*nn+.5*np.mean(norm[list(ns)],axis=0)if ns else nn for nn,ns in zip(norm,neighbors)]);norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-12)
thickness=[];offset=[]
for p,n in zip(v,norm):
 h=tree.ray_cast(Vector(p)-Vector(n)*.025,-Vector(n),20)[0];t=(Vector(p)-h).length if h is not None else 1.0;thickness.append(t);offset.append(max(.05,min(4.1,4.2-t)))
inside=v-norm*.35;outside=v+norm*np.array(offset)[:,None];vv=np.concatenate([inside,outside]);L=len(v);ff=[];edges={}
for f in fv:
 ff.append(tuple(reversed(f)));ff.append(tuple(f+L))
 for a,b in zip(f,np.roll(f,-1)):edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
for es in edges.values():
 if len(es)==1:a,b=es[0];ff.append((a,b,b+L,a+L))
bandmesh=bpy.data.meshes.new('outer wall reinforcement');bandmesh.from_pydata(vv.tolist(),[],ff);bandmesh.update();band=bpy.data.objects.new('outer wall reinforcement',bandmesh);bpy.context.collection.objects.link(band);clean(band);print('BAND',export(band,'outer-band-raw'),flush=True)
np.savez(out/'thickness-samples.npz',v=v,norm=norm,before=np.array(thickness),outward=np.array(offset))
# Keep the existing face-removal envelope and the cleaned neck opening.
op(band,loadpath(src/'printable-seam/thin-lip-removal-tool.stl'),'DIFFERENCE')
def box(c,d):
 bpy.ops.mesh.primitive_cube_add(size=1,location=c);o=bpy.context.object;o.dimensions=d;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
op(band,box((0,0,-74.675),(600,600,250.65)),'DIFFERENCE');n=Vector((-.45,0,1)).normalized();tool=box(Vector((0,0,48.4))-n*300,(1000,1000,600));tool.rotation_euler[1]=-math.atan(.45);op(band,tool,'DIFFERENCE')
print('BAND CUT',export(band,'outer-band'),flush=True);op(head,band,'UNION');print('THICK HEAD',export(head,'thick-head'),flush=True)
(out/'thickening.json').write_text(json.dumps({'targetWallMM':4.2,'direction':'outside; inner optical surface and all mounting datums retained','outerPatchVertices':len(v),'outerPatchFaces':len(fv),'outwardMinMaxMM':[min(offset),max(offset)],'beforePercentiles':np.percentile(thickness,[0,10,50,90,100]).tolist()},indent=2))
