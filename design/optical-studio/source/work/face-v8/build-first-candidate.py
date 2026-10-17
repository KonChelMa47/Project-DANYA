import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v8';out.mkdir(exist_ok=True)
# Reuse mesh utilities only, without running original build.
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v8'
def boolean(a,b,op,remove=True):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
def mesh(name,v,f):
 m=bpy.data.meshes.new(name);m.from_pydata(v,[],f);m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);clean(o);return o
def current(name):
 bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(name+'.stl')));o=bpy.context.object;clean(o);return o
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
v=np.fromfile('work/torso-flat/face-local-v2.positions.bin','<f4').reshape(-1,3).astype(float);v[:,2]+=8
f=np.fromfile('work/torso-flat/face-local-v2.indices.bin','<u4').reshape(-1,3)
of=np.fromfile('work/adult-kit/original-face-outer.indices.bin','<u4').reshape(-1,3)
outer=np.zeros(len(v),bool);outer[np.unique(of)]=True
innerf=f[np.all(~outer[f],axis=1)]
n=np.zeros_like(v);cross=np.cross(v[innerf[:,1]]-v[innerf[:,0]],v[innerf[:,2]]-v[innerf[:,0]])
for j in range(3):np.add.at(n,innerf[:,j],cross)
n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
# Use area weighted inner normals. The closed shell's normals point into cavity.
base=mesh('baseline-screen',v.tolist(),f.tolist());export(base,'screen-before')
w=v.copy();w[~outer]+=.5*n[~outer]
screen=mesh('thicker-screen',w.tolist(),f.tolist());sr=export(screen,'screen-after');assert sr['nonManifoldEdges']==0
np.savez(out/'screen-data.npz',before=v,after=w,faces=f,outerFaces=of,innerFaces=innerf,normals=n)
face=current('face');boolean(face,screen,'UNION',False)
# Original rear wall provides a clearance reference, rather than magnetic cups.
bpy.ops.wm.stl_import(filepath=str(r/'archive-before-integrated-magnets/assembly/head.stl'));head=bpy.context.object
wall=BVHTree.FromPolygons([p.co for p in head.data.vertices],[p.vertices[:]for p in head.data.polygons])
fw=BVHTree.FromPolygons(w.tolist(),f.tolist())
fit=np.load('work/seam_fit.npz');seams=np.load('work/buildable-bust-v10/seam.npy')
def seam(y,z):
 y=y/1.1/100;z=(z/1.1-135)/100;return 1.1*sum(c*y**i*z**j for c,(i,j)in zip(fit['coeff'],fit['powers']))
reps=[]
for row in json.load(open('outputs/danya-simulator/assets/buildable-bust-v10/clear-magnet-mounts.json'))['pairs']:
 idx=row['id'];x,y,z=row['center'];z+=8;sg=np.sign(y);sub=seams[seams[:,1]*sg>40];sub=sub[np.argsort(sub[:,2])];N=25;vv=[]
 for layer in [0,1]:
  for i in range(N):
   t=i/(N-1)
   for j in range(N):
    zz=z-10+20*j/(N-1);ey=float(np.interp(zz-8,sub[:,2],sub[:,1]));sx=seam(ey-sg*.7,zz-8);xx=x+3.55+(sx+7-(x+3.55))*t
    hits=[h for tree in [wall,fw] if (h:=tree.ray_cast(Vector((xx,0,zz)),Vector((0,sg,0)),100)[0]) is not None]
    wy=min(abs(h.y)for h in hits) if hits else abs(ey)-1
    # Start intersects cup's front wall; conformal arm stays inside cavity.
    start=abs(y)+.15
    yy=start*(1-min(t/.18,1))+(wy-.25)*min(t/.18,1)
    # In the forward 5 mm bury outer face 0.2 mm into screen inner wall.
    bury=max(0,min(1,(xx-sx-1)/3))*.5
    yy+=bury
    if layer==0:yy-=2.0
    vv.append((xx,sg*yy,zz))
 arm=grid(vv,N);reps.append(export(arm,'reinforcement-'+str(idx)));boolean(face,arm,'UNION',False)
rep=export(face,'face');(out/'geometry-check.json').write_text(json.dumps({'face':rep,'screen':sr,'reinforcements':reps,'innerNormalOffsetMM':.5,'outerVertexMovementMM':float(np.max(np.linalg.norm(w[outer]-v[outer],axis=1))),'armNominalThicknessMM':2,'armHeightMM':20,'physicalStrengthTest':False},indent=2));print(rep,flush=True)
assert rep['components']==1 and rep['nonManifoldEdges']==0
