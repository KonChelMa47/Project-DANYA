exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=root/'outputs/danya-simulator/assets/projection-prototype/split-head-v6';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(src/'archive-before-printable-mounts-v5/assembly/head.stl'));head=bpy.context.object;wall=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons]);rows=[];C=Vector((-22,0,165))
for deg in [103,123,240,257]:
 d=Vector((math.cos(math.radians(deg)),math.sin(math.radians(deg)),0));t=Vector((-d.y,d.x,0));hit=wall.ray_cast(C,d,200)[0]
 if hit is None:continue
 radius=(hit-C).dot(d);us=sorted(list(np.linspace(-13,13,27))+[-9.3,-8.3,8.3,9.3]);zs=sorted(list(np.linspace(145,187,27))+[156.95,157,175,175.05]);n=len(us);assert len(zs)==n;vs=[]
 for layer in [0,1]:
  for u in us:
   for z in zs:
    start=C+t*float(u);start.z=float(z);h=wall.ray_cast(start,d,200)[0];assert h is not None;rr=(h-start).dot(d)
    vertical=max(0,min(1,(z-145)/12,(187-z)/12));side=max(0,min(1,(abs(u)-8.3)/1));f=1 if 157<=z<=175 else vertical*side
    inside=(1-f)*(rr-2)+f*(radius-14);vs.append(start+d*(inside if layer==0 else rr+.35))
 o=grid(vs,n);rep=export(o,f'flange-{deg}');bpy.data.objects.remove(o,do_unlink=True);assert rep['components']==1 and rep['nonManifoldEdges']==0,rep
 center=C+d*(radius-7);rows.append({'angle':deg,'center':list(center),'radial':list(d),'tangent':list(t),'mesh':rep,'widthMM':26,'depthMM':14,'lowerThicknessMM':8,'upperThicknessMM':10,'gussetPlateauWidthMM':3.7,'wallLinerThicknessMM':2,'wallAttachmentHeightMM':42});print('CANDIDATE',deg,rep,flush=True)
(out/'candidates.json').write_text(json.dumps(rows,indent=2));print(rows)
