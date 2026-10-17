import bpy,bmesh,math,json
from pathlib import Path
from mathutils import Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
def boolean(a,b,op,remove=True):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name)
 if remove:bpy.data.objects.remove(b,do_unlink=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/'face-finished.stl'));face=bpy.context.object
bm=bmesh.new();bm.from_mesh(face.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5)
# Remove the two zero-thickness dangling triangles, then collapse the local folded edge.
dangling=set(f for e in bm.edges if e.is_boundary for f in e.link_faces)
assert len(dangling)==2
bmesh.ops.delete(bm,geom=list(dangling),context='FACES_ONLY')
wire=[e for e in bm.edges if not e.link_faces]
bmesh.ops.delete(bm,geom=wire,context='EDGES')
bad=[e for e in bm.edges if not e.is_manifold]
print('after dangling',len(bad),flush=True)
assert len(bad)==1 and len(bad[0].link_faces)==4
edge=bad[0];mid=sum((v.co for v in edge.verts),Vector())/2
bmesh.ops.pointmerge(bm,verts=list(edge.verts),merge_co=mid)
bmesh.ops.recalc_face_normals(bm,faces=bm.faces)
bad=sum(not e.is_manifold for e in bm.edges)
print('patched',bad,flush=True)
bm.to_mesh(face.data);bm.free()
export(face,'face-local-patched')
assert bad==0
for row in json.load(open(out/'positions.json')):
 x,y,z=row['center'];boolean(face,cylinder((x+2.3,y,z),4.2,4.2),'UNION');boolean(face,cylinder((x+1.75,y,z),3.075,3.3),'DIFFERENCE');boolean(face,cylinder((x+1.8,y,z),3.375,1),'DIFFERENCE')
 for angle in [0,120,240]:boolean(face,cylinder((x+3.375,y+1.6*math.cos(math.radians(angle)),z+1.6*math.sin(math.radians(angle))),.55,.25),'UNION')
 print('pocket restored',row['id'],flush=True)
for n in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(out/(n+'-finished.stl')));boolean(face,bpy.context.object,'DIFFERENCE')
rep=export(face,'face-final');(out/'final-face-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True);assert rep['components']==1 and rep['nonManifoldEdges']==0
