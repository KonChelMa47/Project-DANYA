exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
v=src/'split-head-v6';trees=[]
for path in [src/'archive-before-printable-mounts-v5/assembly/head.stl',v/'head.stl']:
 bpy.ops.wm.stl_import(filepath=str(path));o=bpy.context.object;trees.append(BVHTree.FromPolygons([vv.co for vv in o.data.vertices],[p.vertices[:]for p in o.data.polygons]))
diffs=[];missing=0
for z in np.arange(52,274,2):
 for theta in np.linspace(math.pi/2,3*math.pi/2,181):
  d=Vector((math.cos(theta),math.sin(theta),0));origin=Vector((-22,0,float(z)))+d*200;hits=[t.ray_cast(origin,-d,400)[0]for t in trees]
  if hits[0]is not None and hits[1]is not None:diffs.append((hits[0]-hits[1]).length)
  elif (hits[0]is None)!=(hits[1]is None):missing+=1
report={'rearExteriorRaySamples':len(diffs),'maxSurfaceDeviationMM':max(diffs),'missingHitMismatch':missing,'scope':'Rear hemisphere sampled at2mm height and1degree azimuth. Source is v3 smooth-neck integrated-mirror head before v5 exterior screw holes.'};(v/'exterior-check.json').write_text(json.dumps(report,indent=2));print(report);assert max(diffs)<.002 and missing==0
