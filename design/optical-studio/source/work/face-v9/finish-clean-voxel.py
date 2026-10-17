import bpy,bmesh,json,math
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
def boolean(a,b,op):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);assert not a.modifiers;bpy.data.objects.remove(b,do_unlink=True)
def join(obs):
 bpy.ops.object.select_all(action='DESELECT')
 for o in obs:o.select_set(True)
 bpy.context.view_layer.objects.active=obs[0];bpy.ops.object.join();return bpy.context.object
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/'face-clean-voxel.stl'));face=bpy.context.object
bm=bmesh.new();bm.from_mesh(face.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.0002);u=set(bm.verts);groups=[]
while u:
 st=[u.pop()];vs=set(st)
 while st:
  for e in st.pop().link_edges:
   for v in e.verts:
    if v in u:u.remove(v);vs.add(v);st.append(v)
 groups.append(vs)
groups.sort(key=len,reverse=True);removed=[]
for vs in groups[1:]:
 ext=[max(v.co[i]for v in vs)-min(v.co[i]for v in vs)for i in range(3)];removed.append({'vertices':len(vs),'extents':ext});assert len(vs)<500 and max(ext)<2,removed;bmesh.ops.delete(bm,geom=list(vs),context='VERTS')
bm.to_mesh(face.data);bm.free();
rows=json.load(open(out/'positions.json'))
for kind,op in [('cup','UNION'),('bore','DIFFERENCE'),('key','DIFFERENCE'),('lands','UNION')]:
 obs=[]
 for row in rows:
  x,y,z=row['center']
  if kind=='cup':obs.append(cylinder((x+2.3,y,z),4.2,4.2))
  elif kind=='bore':obs.append(cylinder((x+1.75,y,z),3.075,3.3))
  elif kind=='key':obs.append(cylinder((x+1.8,y,z),3.375,1))
  else:
   for a in [0,120,240]:obs.append(cylinder((x+3.375,y+1.6*math.cos(math.radians(a)),z+1.6*math.sin(math.radians(a))),.55,.25))
 boolean(face,join(obs),op);print('restored',kind,flush=True)
for n in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(out/(n+'-finished.stl')));boolean(face,bpy.context.object,'DIFFERENCE');print('trimmed',n,flush=True)
bm=bmesh.new();bm.from_mesh(face.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.0002);u=set(bm.verts);groups=[]
while u:
 st=[u.pop()];vs=set(st)
 while st:
  for e in st.pop().link_edges:
   for v in e.verts:
    if v in u:u.remove(v);vs.add(v);st.append(v)
 groups.append(vs)
groups.sort(key=len,reverse=True);removed=[]
for vs in groups[1:]:
 ext=[max(v.co[i]for v in vs)-min(v.co[i]for v in vs)for i in range(3)];removed.append({'vertices':len(vs),'extents':ext});assert len(vs)<500 and max(ext)<2,removed;bmesh.ops.delete(bm,geom=list(vs),context='VERTS')
bm.to_mesh(face.data);bm.free();rep=export(face,'face-clean-final');rep['removedBooleanSlivers']=removed;(out/'clean-final-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True);assert rep['components']==1 and rep['nonManifoldEdges']==0
