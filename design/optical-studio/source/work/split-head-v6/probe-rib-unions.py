exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=root/'outputs/danya-simulator/assets/projection-prototype/split-head-v6';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(src/'archive-before-printable-mounts-v5/assembly/head.stl'));head=bpy.context.object;wall=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons]);rows=[];C=Vector((-22,0,165))
for deg in [105,115,125,135,225,235,245,255]:
 d=Vector((math.cos(math.radians(deg)),math.sin(math.radians(deg)),0));t=Vector((-d.y,d.x,0));hit=wall.ray_cast(C,d,200)[0]
 if hit is None:continue
 radius=(hit-C).dot(d);n=9
 def patch(u0,u1,z0,z1,depth_fn):
  vs=[]
  for layer in [0,1]:
   for i in range(n):
    for j in range(n):
     u=u0+(u1-u0)*i/(n-1);z=z0+(z1-z0)*j/(n-1);start=C+t*u;start.z=z;h=wall.ray_cast(start,d,200)[0];assert h is not None;rr=(h-start).dot(d);inside=depth_fn(z,rr);vs.append(start+d*(inside if layer==0 else rr+.35))
  return grid(vs,n)
 o=patch(-11,11,157,175,lambda z,rr:radius-14)
 for u0,u1 in [(-11,-8.3),(8.3,11)]:
  for z0,z1 in [(145,157.1),(174.9,187)]:
   rib=patch(u0,u1,z0,z1,lambda z,rr:rr-(2+12*max(0,min(1,(z-145)/12,(187-z)/12))))
   boolean(o,rib,'UNION')
 for _ in range(3):clean(o)
 rep=export(o,f'flange-{deg}');bpy.data.objects.remove(o,do_unlink=True);print('CANDIDATE',deg,rep,flush=True)
 if rep['components']!=1 or rep['nonManifoldEdges']!=0:continue
 center=C+d*(radius-7);rows.append({'angle':deg,'center':list(center),'radial':list(d),'tangent':list(t),'mesh':rep,'widthMM':22,'depthMM':14,'lowerThicknessMM':8,'upperThicknessMM':10,'gussetThicknessMM':2.7,'wallAttachmentHeightMM':42})
(out/'candidates.json').write_text(json.dumps(rows,indent=2));print(rows)
