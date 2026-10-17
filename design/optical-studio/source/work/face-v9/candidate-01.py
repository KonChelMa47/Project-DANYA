import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9';out.mkdir(exist_ok=True)
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
def boolean(a,b,op,remove=True):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
def triangles(o):
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.triangulate(bm,faces=list(bm.faces),quad_method='FIXED',ngon_method='EAR_CLIP');bm.to_mesh(o.data);bm.free();return o
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
trees=[]
for file in ['archive-before-integrated-magnets/assembly/head.stl','face-v8/screen-after.stl']:
 bpy.ops.wm.stl_import(filepath=str(r/file));o=bpy.context.object;trees.append(BVHTree.FromPolygons([v.co for v in o.data.vertices],[p.vertices[:]for p in o.data.polygons]))
seams=np.load('work/buildable-bust-v10/seam.npy');seams[:,2]+=8
fit=np.load('work/seam_fit.npz')
def seam(y,z):
 y=y/1.1/100;z=((z-8)/1.1-135)/100;return 1.1*sum(c*y**i*z**j for c,(i,j)in zip(fit['coeff'],fit['powers']))
def edge(y,sg):
 sub=seams[seams[:,2]>210]if sg==1 else seams[seams[:,2]<100];sub=sub[np.argsort(sub[:,1])];z=float(np.interp(y,sub[:,1],sub[:,2]));return seam(y,z),z

def hits(tree,x,y,sg):
 p=Vector((x,y,150));hs=[]
 for _ in range(6):
  h=tree.ray_cast(p,Vector((0,0,sg)),200)[0]
  if h is None:break
  hs.append(h.z);p=h+Vector((0,0,sg*.002))
 return hs
rows=[]
for idx,(sg,y,dx)in enumerate([(1,-28,14),(1,28,14),(-1,-40,12),(-1,40,12)]):
 sx,sz=edge(y,sg);x=sx-dx
 # Fit the entire 8.4mm diameter cup below/above the sloping head wall.
 limit=[]
 for xx in np.linspace(x-4.2,x+4.4,17):
  for yy in np.linspace(y-4.2,y+4.2,17):
   hh=hits(trees[0],xx,yy,sg)
   if hh:limit.append(sg*hh[0]-4.5)
 z=sg*min(limit);N=29;vv=[]
 for layer in [0,1]:
  for i in range(N):
   t=i/(N-1)
   for j in range(N):
    yy=y-7+14*j/(N-1);ex,ez=edge(yy,sg);xx=x+3.55+(ex+1.0-(x+3.55))*t
    hh=hits(trees[0],xx,yy,sg);fh=hits(trees[1],xx,yy,sg)
    if hh:zz=hh[0]-sg*.35
    elif fh:zz=fh[-1]-sg*.25
    else:zz=ez-sg*.6
    # Broad arm joins cup rear face then follows the shell clear of beam.
    zz=z*(1-min(t/.18,1))+zz*min(t/.18,1)
    if layer==0:zz-=sg*2
    vv.append((xx,yy,zz))
 arm=triangles(grid(vv,N));boolean(arm,cylinder((x+2.3,y,z),4.2,4.2),'UNION')
 boolean(arm,cylinder((x+1.75,y,z),3.075,3.3),'DIFFERENCE');boolean(arm,cylinder((x+1.8,y,z),3.375,1),'DIFFERENCE')
 for angle in [0,120,240]:
  boolean(arm,cylinder((x+3.375,y+1.6*math.cos(math.radians(angle)),z+1.6*math.sin(math.radians(angle))),.55,.25),'UNION')
 faceRep=export(arm,'face-holder-'+str(idx));assert faceRep['components']==1 and faceRep['nonManifoldEdges']==0
 # Rear pocket with a broad foot spanning from actual inner shell to the cup.
 vv=[];N=15
 for layer in [0,1]:
  for i in range(N):
   xx=x-5.5+5.3*i/(N-1)
   for j in range(N):
    if sg==1:
     yy=y-5.5+11*j/(N-1);hh=hits(trees[0],xx,yy,sg);assert hh,(idx,xx,yy)
     zz=hh[0]+sg*.25 if layer else z-sg*1.5
    else:
     zz=z-5.5+11*j/(N-1);sign=1 if y>0 else-1
     h=trees[0].ray_cast(Vector((xx,0,zz)),Vector((0,sign,0)),150)[0];assert h,(idx,xx,zz)
     yy=h.y+sign*.25 if layer else y-sign*1.5
    vv.append((xx,yy,zz))
 back=triangles(grid(vv,N));boolean(back,cylinder((x-2.1,y,z),4.2,4.2),'UNION')
 boolean(back,cylinder((x-1.6,y,z),3.075,3.4),'DIFFERENCE');boolean(back,cylinder((x-1.6,y,z),3.375,1),'DIFFERENCE')
 for angle in [0,120,240]:
  boolean(back,cylinder((x-3.225,y+1.6*math.cos(math.radians(angle)),z+1.6*math.sin(math.radians(angle))),.55,.35),'UNION')
 headRep=export(back,'head-holder-'+str(idx));assert headRep['components']==1 and headRep['nonManifoldEdges']==0
 for side,cx in [('face',x+1.8),('head',x-1.6)]:export(cylinder((cx,y,z),2.95,2.9),'magnet-clear-'+str(idx)+'-'+side+'-magnet')
 rows.append({'id':idx,'center':[x,y,z],'top':sg==1,'seam':[sx,y,sz],'face':faceRep,'head':headRep});print('built',idx,rows[-1],flush=True)
(out/'positions.json').write_text(json.dumps(rows,indent=2))
