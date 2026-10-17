# Local inner-surface relief only; preserve every original outer vertex and mesh topology.
import bpy,bmesh,numpy as np,json
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
root=Path.cwd();base=root/'outputs/danya-simulator/assets/torso-flat-study';out=root/'outputs/danya-simulator/assets/buildable-bust-v10'
v=np.fromfile(base/'head.positions.bin',np.float32).reshape(-1,3);f=np.fromfile(base/'head.indices.bin',np.uint32).reshape(-1,3);N=len(v)//2;original=v.copy();normals=np.zeros((N,3),float);outer=f[np.all(f<N,axis=1)]
cross=np.cross(v[outer[:,1]]-v[outer[:,0]],v[outer[:,2]]-v[outer[:,0]])
for j in range(3):np.add.at(normals,outer[:,j],cross)
normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-20)
parts=[]
for i in range(4):
 name=f'magnet-embedded-{i}-face-return';p=np.fromfile(out/(name+'.positions.bin'),np.float32).reshape(-1,3);q=np.fromfile(out/(name+'.indices.bin'),np.uint32).reshape(-1,3);parts.append((name,BVHTree.FromPolygons(p.tolist(),q.tolist(),all_triangles=True),p.min(0)-1,p.max(0)+1))
forcedFile=root/'work/buildable-bust-v10/head-embedded-forced-relief.json';forced=json.load(open(forcedFile)) if forcedFile.exists() else {};changed=[];infeasible=[];clearance=.15;minWall=.45
for i in range(N):
 inner=v[N+i].astype(float);outerPoint=v[i].astype(float);delta=inner-outerPoint;length=np.linalg.norm(delta)
 if length<1e-7:continue
 direction=delta/length;cos=-float(normals[i]@direction)
 if cos<.02:continue
 desired=length
 for name,bvh,lo,hi in parts:
  if np.any(inner<lo)or np.any(inner>hi):continue
  p,n,tri,dist=bvh.ray_cast(Vector(outerPoint),Vector(direction),float(length+1.0))
  if p is not None and dist<desired+clearance:desired=min(desired,float(dist)-clearance)
 desired-=forced.get(str(i),0)
 if desired>=length-1e-5:continue
 remaining=desired*cos;oldWall=length*cos
 if remaining<min(minWall,oldWall)-1e-4:
  infeasible.append({'vertex':int(i),'oldNormalWall':oldWall,'neededNormalWall':remaining,'point':inner.tolist()});continue
 v[N+i]=outerPoint+direction*desired;changed.append({'vertex':int(i),'reliefMM':length-desired,'remainingNormalWallMM':remaining})
r={'status':'candidate, not exported if minimum wall condition fails','outerVerticesUnchanged':bool(np.array_equal(v[:N],original[:N])),'changed':len(changed),'infeasibleCount':len(infeasible),'minimumNormalWallMM':min([x['remainingNormalWallMM']for x in changed],default=None),'maximumReliefMM':max([x['reliefMM']for x in changed],default=0),'infeasible':infeasible[:50]}
if not infeasible:
 m=bpy.data.meshes.new('head-embedded-relief');m.from_pydata(v.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new(m.name,m);bpy.context.collection.objects.link(o);bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o
 bm=bmesh.new();bm.from_mesh(m);r['nonManifoldEdges']=sum(not e.is_manifold for e in bm.edges);r['volume']=bm.calc_volume(signed=False);bm.free();bpy.ops.wm.stl_export(filepath=str(out/'head-embedded-relief.stl'),export_selected_objects=True);v.tofile(out/'head-embedded-relief.positions.bin');f.tofile(out/'head-embedded-relief.indices.bin')
(out/'head-embedded-relief-report.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:val for k,val in r.items()if k!='infeasible'}),flush=True)
