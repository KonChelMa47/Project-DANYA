exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
v=src/'printable-mounts-v5';data=json.load(open(v/'head-check.json'));d=data['mirrorMatrix'];m=Matrix([[d[i+4*j]for j in range(4)]for i in range(4)]);inv=m.inverted()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
objs={}
for n in ['head','mirror-removable-cradle']:
 bpy.ops.wm.stl_import(filepath=str(v/(n+'.stl')));o=bpy.context.object;clean(o);objs[n]=o
 for vert in o.data.vertices:vert.co=inv@vert.co
head=objs['head'];cradle=objs['mirror-removable-cradle'];trees={n:BVHTree.FromPolygons([v.co for v in o.data.vertices],[p.vertices[:]for p in o.data.polygons])for n,o in objs.items()};rows=[]
for a in data['anchors']:
 x,y=a['xy'];hvals=[];nvals=[]
 for theta in np.arange(0,math.tau,math.tau/24):
  for radius in [2.0,2.6,3.4]:
   hit=trees['head'].ray_cast(Vector((x+radius*math.cos(theta),y+radius*math.sin(theta),-80)),Vector((0,0,1)),60)[0];assert hit is not None;hvals.append(hit.z)
  hit=trees['mirror-removable-cradle'].ray_cast(Vector((x+2.4*math.cos(theta),y+2.4*math.sin(theta),-30)),Vector((0,0,-1)),15)[0];assert hit is not None;nvals.append(hit.z)
 rows.append({'xy':[x,y],'washerSeatMaxErrorMM':max(abs(z+44)for z in hvals),'nutShoulderMaxErrorMM':max(abs(z+32.6)for z in nvals)})
# Closed-volume common of neutral head and cartridge; coplanar mount contact permitted.
copy=head.copy();copy.data=head.data.copy();bpy.context.collection.objects.link(copy);boolean(copy,cradle,'INTERSECT',remove=False);clean(copy);bm=bmesh.new();bm.from_mesh(copy.data);overlap=abs(bm.calc_volume());bad=sum(not e.is_manifold for e in bm.edges);bounds=[[min(v.co[i]for v in bm.verts),max(v.co[i]for v in bm.verts)]for i in range(3)]if bm.verts else [];bm.free();print('COMMON',bad,bounds,flush=True)
rawThickness=bounds[2][1]-bounds[2][0]if bounds else 0
assert rawThickness<.0001,('Unexpected bulk common',bounds)
# STL roundoff creates open coplanar slivers. A0.01mm normal offset distinguishes intended contact from bulk penetration.
for vert in cradle.data.vertices:vert.co.z+=.01
copy2=head.copy();copy2.data=head.data.copy();bpy.context.collection.objects.link(copy2);boolean(copy2,cradle,'INTERSECT',remove=False);clean(copy2);bm=bmesh.new();bm.from_mesh(copy2.data);overlap=abs(bm.calc_volume());bm.free()
report={'anchors':rows,'headCradleOverlapMM3':overlap,'verificationNormalOffsetMM':.01,'unshiftedCommonThicknessMM':rawThickness,'unshiftedCommonIsCoplanarRoundoff':True,'preloadPath':'Screw head -> washer -> rear-shell boss -> removable cradle shoulder -> front-loaded nut','physicalStrengthTest':False}
(v/'head-mating-check.json').write_text(json.dumps(report,indent=2));print(report)
assert overlap<.01 and all(max(x['washerSeatMaxErrorMM'],x['nutShoulderMaxErrorMM'])<.002 for x in rows)
