exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=root/'outputs/danya-simulator/assets/projection-prototype/split-head-v6';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(src/'archive-before-printable-mounts-v5/assembly/head.stl'));head=bpy.context.object;wall=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons]);rows=[];C=Vector((-22,0,165))
for deg in range(100,261,10):
 d=Vector((math.cos(math.radians(deg)),math.sin(math.radians(deg)),0));t=Vector((-d.y,d.x,0));hit=wall.ray_cast(C,d,200)[0]
 if hit is None:continue
 radius=(hit-C).dot(d);vs=[];n=9;coords=[]
 for layer in [0,1]:
  for i in range(n):
   for j in range(n):
    u=-8+16*i/(n-1);z=159+13*j/(n-1);start=C+t*u;start.z=z;h=wall.ray_cast(start,d,200)[0];assert h is not None
    vs.append(start+d*(radius-12 if layer==0 else(h-start).dot(d)+.35))
 o=grid(vs,n);rep=export(o,f'flange-{deg}');bpy.data.objects.remove(o,do_unlink=True)
 center=C+d*(radius-6.5);rows.append({'angle':deg,'center':list(center),'radial':list(d),'tangent':list(t),'mesh':rep})
(out/'candidates.json').write_text(json.dumps(rows,indent=2));print(rows)
