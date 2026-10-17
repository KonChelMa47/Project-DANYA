import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
root=Path.cwd();src=root/'outputs/danya-simulator/assets/projection-prototype';out=src/'magnets-v2';out.mkdir(exist_ok=True)
fit=np.load(root/'work/seam_fit.npz');seams=np.load(root/'work/buildable-bust-v10/seam.npy')
def seam(y,z):
 y=y/1.1/100;z=(z/1.1-135)/100;return 1.1*sum(c*y**i*z**j for c,(i,j)in zip(fit['coeff'],fit['powers']))
def clean(o):
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5)
 # Weld collinear Boolean T-junctions without filling or moving any surface.
 for attempt in range(32):
  edges=[e for e in bm.edges if e.is_boundary];verts={v for e in edges for v in e.verts};found=False
  for e in edges:
   a,b=e.verts;d=b.co-a.co
   if d.length_squared<1e-12:continue
   for v in verts:
    if v in (a,b):continue
    t=(v.co-a.co).dot(d)/d.length_squared
    if .0001<t<.9999 and (a.co+d*t-v.co).length<1e-5:
     bmesh.utils.edge_split(e,a,t);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5);found=True;break
   if found:break
  if not found:break
 bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(o.data);bm.free()
def load(n):
 p=src/'archive-before-integrated-magnets'/'assembly'/(n+'.stl');p=p if p.exists()else src/'assembly'/(n+'.stl');bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;o.name=n;clean(o);return o
def boolean(a,b,op,remove=True):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='EXACT';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
def cylinder(center,r,L):
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=r,depth=L,location=center,rotation=(0,math.pi/2,0));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
def grid(v,n):
 faces=[];N=n*n
 for l in [0,1]:
  for i in range(n-1):
   for j in range(n-1):a=l*N+i*n+j;faces.append((a,a+1,a+n+1,a+n))
 edge=list(range(n))+[i*n+n-1 for i in range(1,n)]+[(n-1)*n+j for j in range(n-2,-1,-1)]+[i*n for i in range(n-2,0,-1)]
 for a,b in zip(edge,edge[1:]+edge[:1]):faces.append((a,b,b+N,a+N))
 me=bpy.data.meshes.new('bridge');me.from_pydata(v,[],faces);me.update();o=bpy.data.objects.new('bridge',me);bpy.context.collection.objects.link(o);clean(o);return o
def export(o,n):
 clean(o);bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(out/(n+'.stl')),export_selected_objects=True)
 bm=bmesh.new();bm.from_mesh(o.data);unseen=set(bm.verts);components=0
 while unseen:
  components+=1;stack=[unseen.pop()]
  while stack:
   v=stack.pop()
   for e in v.link_edges:
    w=e.other_vert(v)
    if w in unseen:unseen.remove(w);stack.append(w)
 report={'components':components,'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'volumeMM3':bm.calc_volume(signed=False)};bm.free();return report
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
head=load('head');face=load('face');wall=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons]);rows=[]
for row in json.load(open(root/'outputs/danya-simulator/assets/buildable-bust-v10/clear-magnet-mounts.json'))['pairs']:
 idx=row['id'];x,y,z=row['center'];z+=8;sign=1 if y>0 else-1;parts={}
 for side,suffix in [('face','face-return'),('head','head-cup')]:
  holder=load(f'magnet-clear-{idx}-{suffix}')
  # Epoxy undercut:1mm wide,diameter6.75; preserves>=0.825mm cup outer wall.
  cx=x+(1.8 if side=='face'else-1.6);boolean(holder,cylinder((cx,y,z),3.375,1),'DIFFERENCE')
  # Three small lands maintain magnet face positions with room for epoxy underneath.
  for angle in [0,120,240]:
   yy=y+1.6*math.cos(math.radians(angle));zz=z+1.6*math.sin(math.radians(angle));cx=x+(3.375 if side=='face'else-3.225);boolean(holder,cylinder((cx,yy,zz),.55,.25 if side=='face'else.35),'UNION')
  n=11;v=[]
  if side=='face':
   sub=seams[seams[:,1]*sign>40];sub=sub[np.argsort(sub[:,2])]
   for u in [-.25,.2]:
    for i in range(n):
     for j in range(n):
      zz=z-10+20*j/(n-1);edgeY=float(np.interp(zz-8,sub[:,2],sub[:,1]));yy=edgeY-sign*(.4+.6*i/(n-1));v.append((seam(yy,zz-8)+u,yy,zz))
  else:
   for layer in [0,1]:
    for i in range(n):
     xx=x-5.4+5.1*i/(n-1)
     for j in range(n):
      zz=z-5.9+11.8*j/(n-1);h=wall.ray_cast(Vector((xx,0,zz)),Vector((0,sign,0)),100)[0]
      assert h is not None
      yy=h.y+sign*.25 if layer else h.y-sign*.35;v.append((xx,yy,zz))
  boolean(holder,grid(v,n),'UNION');rep=export(holder,f'integrated-{idx}-{side}-addition');print('ADDITION',idx,side,rep,flush=True);assert rep['components']==1 and rep['nonManifoldEdges']==0,rep
  # Preserve additive part for contact/optical delta checks, fuse into appropriate shell.
  boolean(face if side=='face'else head,holder,'UNION',remove=False);parts[side]=rep
 rows.append({'id':idx,'parts':parts});print('PAIR',idx,flush=True)
reports={n:export(o,n+'-integrated')for n,o in [('head',head),('face',face)]}
(out/'integration-report.json').write_text(json.dumps({'shells':reports,'pairs':rows,'retention':'two-part epoxy, internal annular adhesive key and three axial seating lands','magnetMM':[5.9,2.9],'pocketDiameterMM':6.15,'undercutDiameterMM':6.75,'undercutWidthMM':1,'magnetFaceGapMM':.5,'exteriorSeamOffsetMM':0,'physicalPullTest':False},indent=2));print('RESULT',reports,flush=True)
assert all(r['components']==1 and r['nonManifoldEdges']==0 for r in reports.values())
