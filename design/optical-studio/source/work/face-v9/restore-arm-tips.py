import bpy,bmesh,json,math,numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
def load(n):
 bpy.ops.wm.stl_import(filepath=str(out/(n+'.stl')));return bpy.context.object
def boolean(a,b,op):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);assert not a.modifiers;bpy.data.objects.remove(b,do_unlink=True)
def box(lo,hi):
 bpy.ops.mesh.primitive_cube_add(size=1,location=[(a+b)/2 for a,b in zip(lo,hi)]);o=bpy.context.object;o.dimensions=[b-a for a,b in zip(lo,hi)];bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);face=load('face-clean-voxel');bm=bmesh.new();bm.from_mesh(face.data);u=set(bm.verts);groups=[]
while u:
 st=[u.pop()];vs=set(st)
 while st:
  for e in st.pop().link_edges:
   for v in e.verts:
    if v in u:u.remove(v);vs.add(v);st.append(v)
 groups.append(vs)
groups.sort(key=len,reverse=True)
for vs in groups[1:]:
 assert len(vs)<500 and max(max(v.co[i]for v in vs)-min(v.co[i]for v in vs)for i in range(3))<2
 bmesh.ops.delete(bm,geom=list(vs),context='VERTS')
# Smooth only the few adaptive-tessellation folds, under 0.6mm in radius.
reports=[json.loads(l) for l in Path('work/face-v9/all-clean-crossings.jsonl').read_text().splitlines() if l.startswith('{')];bm.verts.ensure_lookup_table();kd=KDTree(len(bm.verts))
for i,v in enumerate(bm.verts):kd.insert(v.co,i)
kd.balance();sel=set()
for e in reports[0]['examples']:
 for p in e['a']+e['b']:
  for co,i,d in kd.find_range(Vector(p),.6):sel.add(bm.verts[i])
for _ in range(12):
 moves={v:sum((e.other_vert(v).co for e in v.link_edges),Vector())/len(v.link_edges)for v in sel}
 for v,p in moves.items():v.co=v.co.lerp(p,.15)
# Match the nominal curved rear division surface, without subtracting the full head mesh.
fit=np.load('work/seam_fit.npz')
def seam(y,z):
 y/=110;z=((z-8)/1.1-135)/100;return 1.1*sum(c*y**i*z**j for c,(i,j)in zip(fit['coeff'],fit['powers']))
bm.normal_update();snaps=0
for v in bm.verts:
 sx=seam(v.co.y,v.co.z)-6
 if abs(v.co.x-sx)<.23 and v.normal.x<-.65:v.co.x=sx;snaps+=1
bm.to_mesh(face.data);bm.free();print('cleaned base',len(sel),snaps,flush=True);print(export(face,'face-clean-base'),flush=True)
for row in json.load(open(out/'positions.json')):
 i=row['id'];x,y,z=row['center'];arm=load(f'face-holder-{i}')
 # Replace the entire cup and its short starting segment, preserving the analytic hole/adhesive key.
 boolean(arm,box((x-6,y-12,z-14),(x+4.6,y+12,z+14)),'INTERSECT')
 boolean(face,box((x-6,y-12,z-14),(x+4.3,y+12,z+14)),'DIFFERENCE')
 boolean(face,arm,'UNION');print('analytic tip',i,flush=True)
rep=export(face,'face-tips-restored');(out/'tips-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
