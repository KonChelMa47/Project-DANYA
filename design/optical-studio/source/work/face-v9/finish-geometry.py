import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
def boolean(a,b,op,remove=True):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
def load(p):
 bpy.ops.wm.stl_import(filepath=str(p));return bpy.context.object
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
heads=[];reports={}
for name in ['head-upper','head-lower']:
 o=load(out/(name+'.stl'));bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5);unseen=set(bm.verts);groups=[]
 while unseen:
  st=[unseen.pop()];vs=set(st)
  while st:
   for e in st.pop().link_edges:
    for v in e.verts:
     if v in unseen:unseen.remove(v);vs.add(v);st.append(v)
  groups.append(vs)
 groups.sort(key=len,reverse=True)
 for vs in groups[1:]:
  extent=max(max(v.co[i]for v in vs)-min(v.co[i]for v in vs)for i in range(3));assert len(vs)<200 and extent<8 and min(max(v.co[i]for v in vs)-min(v.co[i]for v in vs)for i in range(3))<.21,(name,len(vs),extent)
  bmesh.ops.delete(bm,geom=list(vs),context='VERTS')
 bm.to_mesh(o.data);bm.free();clean(o);rep=export(o,name+'-finished');reports[name]=rep;print(name,rep,flush=True);assert rep['nonManifoldEdges']==0 and rep['components']==1;heads.append(o)
face=load(out/'face-repaired.stl');bpy.context.view_layer.objects.active=face;m=face.modifiers.new('Adaptive printable tessellation','DECIMATE');m.ratio=.06;bpy.ops.object.modifier_apply(modifier=m.name);print('decimated',len(face.data.vertices),flush=True)
a=np.load(r/'face-v8/screen-data.npz');outer=BVHTree.FromPolygons(a['before'].tolist(),a['outerFaces'].tolist());inner=BVHTree.FromPolygons(a['after'].tolist(),a['innerFaces'].tolist())
originalHead=load(r/'archive-before-integrated-magnets/assembly/head.stl');ht=BVHTree.FromPolygons([v.co for v in originalHead.data.vertices],[p.vertices[:]for p in originalHead.data.polygons])
fit=np.load('work/seam_fit.npz')
def seam(y,z):
 y=y/1.1/100;z=((z-8)/1.1-135)/100;return 1.1*sum(c*y**i*z**j for c,(i,j)in zip(fit['coeff'],fit['powers']))
bm=bmesh.new();bm.from_mesh(face.data);bm.normal_update();stats={'outerSnaps':0,'collarSnaps':0,'innerThicknessCorrections':0}
for v in bm.verts:
 p=v.co;h,n,_,d=outer.find_nearest(p)
 if d<.35 and n.dot(v.normal)>.6:v.co=h;stats['outerSnaps']+=1;continue
 if p.x>=seam(p.y,p.z)-6.1 and p.x<seam(p.y,p.z)+.5:
  h,n,_,d=ht.find_nearest(p)
  if d<.3 and n.dot(v.normal)>.6:v.co=h;stats['collarSnaps']+=1
  if abs(p.x-(seam(p.y,p.z)-6))<.25 and v.normal.x<-.45:v.co.x=seam(p.y,p.z)-6
 # Keep the original v8 minimum local interior envelope; never make it thinner by voxel rounding.
 h,n,_,d=inner.find_nearest(v.co)
 if d<.3 and n.dot(v.normal)>.6 and (v.co-h).dot(n)<0:v.co=h+n*.015;stats['innerThicknessCorrections']+=1
bm.to_mesh(face.data);bm.free();clean(face)
for row in json.load(open(out/'positions.json')):
 x,y,z=row['center'];boolean(face,cylinder((x+2.3,y,z),4.2,4.2),'UNION');boolean(face,cylinder((x+1.75,y,z),3.075,3.3),'DIFFERENCE');boolean(face,cylinder((x+1.8,y,z),3.375,1),'DIFFERENCE')
 for angle in [0,120,240]:boolean(face,cylinder((x+3.375,y+1.6*math.cos(math.radians(angle)),z+1.6*math.sin(math.radians(angle))),.55,.25),'UNION')
# Trim voxel rounding at the exact unchanged rear-shell interface.
for head in heads:boolean(face,head,'DIFFERENCE',False)
rep=export(face,'face-finished');reports['face']=rep;reports['reconstruction']=stats;(out/'finished-geometry-check.json').write_text(json.dumps(reports,indent=2));print(reports,flush=True);assert rep['components']==1 and rep['nonManifoldEdges']==0
