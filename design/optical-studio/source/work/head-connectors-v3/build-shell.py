exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=root/'outputs/danya-simulator/assets/head-connectors-v3'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def load_current(name):
 bpy.ops.wm.stl_import(filepath=str(src/'assembly'/(name+'.stl')));o=bpy.context.object;o.name=name;clean(o);return o
head=load_current('head');wall=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons]);original=head.copy();original.data=head.data.copy()
# Smooth radial interior trimming tool. Exterior is sampled and retained with>=1.3mm radial clearance.
N=360;zs=np.arange(49.5,62.01,.25);vs=[];cx,cy=-22.0161686,.3457451;th=np.arange(N)*math.tau/N
A=np.column_stack([np.ones(N)]+[f(k*th)for k in range(1,13)for f in (np.cos,np.sin)])
for z in zs:
 radii=[]
 for t in th:
  d=Vector((math.cos(t),math.sin(t),0));h=wall.ray_cast(Vector((cx,cy,max(z,50.8)))+d*100,-d,100)[0];radii.append((h-Vector((cx,cy,max(z,50.8)))).length if h is not None else np.nan)
 rr=np.array(radii);valid=np.isfinite(rr);pred=np.sum(A*np.linalg.lstsq(A[valid],rr[valid],rcond=None)[0][None,:],axis=1)
 # Fit gently, but never cut nearer than1.3mm to a measured outside surface.
 pred=np.where(valid,np.minimum(pred,rr),pred);thickness=float(np.interp(z,[49.5,51,54,58,60,62],[2.2,2.2,1.3,1.3,2.5,7]));radius=pred-thickness
 for j,t in enumerate(th):vs.append((cx+radius[j]*math.cos(t),cy+radius[j]*math.sin(t),z))
fs=[]
for k in range(len(zs)-1):
 for j in range(N):a=k*N+j;b=k*N+(j+1)%N;fs.append((a,b,b+N,a+N))
fs.extend([tuple(reversed(range(N))),tuple((len(zs)-1)*N+j for j in range(N))]);me=bpy.data.meshes.new('neck-inner-tool');me.from_pydata(vs,[],fs);me.update();tool=bpy.data.objects.new('neck-inner-tool',me);bpy.context.collection.objects.link(tool);clean(tool);boolean(head,tool,'DIFFERENCE');print('NECK INNER DONE',flush=True)
bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,-49.35));tool=bpy.context.object;tool.dimensions=(500,500,200);bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(head,tool,'DIFFERENCE')
# Delete disconnected scraps from the trimmed front lip; retain the complete shell.
bm=bmesh.new();bm.from_mesh(head.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);unseen=set(bm.verts);components=[]
while unseen:
 stack=[unseen.pop()];group=set(stack)
 while stack:
  vv=stack.pop()
  for edge in vv.link_edges:
   w=edge.other_vert(vv)
   if w in unseen:unseen.remove(w);group.add(w);stack.append(w)
 components.append(group)
components.sort(key=len,reverse=True);scraps=[vv for group in components[1:]for vv in group];assert len(scraps)<500
bmesh.ops.delete(bm,geom=scraps,context='VERTS');bm.to_mesh(head.data);bm.free()
report={'neck':export(head,'head-neck-clean'),'neckTrimPlaneMM':50.65,'removedDisconnectedScrapVertices':len(scraps)};print('NECK',report,flush=True)
# Mirror support is now part of head. Bridge the previous adhesive gap with conformal patches.
d=json.load(open(root/'work/buildable-bust-v8/datums.json'))['mirror']['matrix'];d[14]+=8;m=Matrix([[d[i+4*j]for j in range(4)]for i in range(4)]);inv=m.inverted();direction=m.to_3x3()@Vector((0,0,-1));cradle=load_current('mirror-head-cradle')
for x in [-16,16]:
 v=[];n=13
 for layer in [0,1]:
  for i in range(n):
   for j in range(n):
    xx=x-6.8+13.6*i/(n-1);yy=-11.8+23.6*j/(n-1);hit=wall.ray_cast(m@Vector((xx,yy,-24)),direction,100)[0];assert hit is not None;h=inv@hit;v.append(m@Vector((xx,yy,h.z+(.4 if layer==0 else-.25))))
 boolean(cradle,grid(v,n),'UNION')
report['mirrorAddition']=export(cradle,'mirror-integrated-addition');boolean(head,cradle,'UNION');print('MIRROR DONE',flush=True)
# Captive-nut mounting bosses, matching bolt-on static and motorized supports.
for p in json.load(open(out/'connectors.json'))['interfaces']:
 d=p['matrix'];m=Matrix([[d[i+4*j]for j in range(4)]for i in range(4)]);inv=m.inverted();normal=m.to_3x3()@Vector((0,0,1));v=[];n=13;depths=[]
 for layer in [0,1]:
  for i in range(n):
   for j in range(n):
    x=-9+18*i/(n-1);y=-6+12*j/(n-1);h=wall.ray_cast(m@Vector((x,y,-10)),normal,30)[0];assert h is not None,(p['id'],x,y);z=(inv@h).z;depths.append(z);v.append(m@Vector((x,y,-5 if layer==0 else z+.25)))
 boss=grid(v,n);boolean(head,boss,'UNION')
 # Regular hex with across-flats5.8mm, rear-entry pocket.
 bpy.ops.mesh.primitive_cylinder_add(vertices=6,radius=5.8/math.sqrt(3),depth=2.75,location=(0,0,-3.725),rotation=(0,0,math.pi/6));tool=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 for vert in tool.data.vertices:vert.co=m@vert.co
 boolean(head,tool,'DIFFERENCE')
 bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=1.7,depth=3.8,location=(0,0,-3.2));tool=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 for vert in tool.data.vertices:vert.co=m@vert.co
 boolean(head,tool,'DIFFERENCE');print('BOSS',p['id'],min(depths),max(depths),flush=True)
report['head']=export(head,'head-integrated');(out/'shell-report.json').write_text(json.dumps(report,indent=2));print('REPORT',report,flush=True);assert report['head']['components']==1 and report['head']['nonManifoldEdges']==0
