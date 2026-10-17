import bpy,bmesh,numpy as np,json,time
from pathlib import Path
from mathutils.kdtree import KDTree
root=Path(__file__).resolve().parent.parent;start=time.time()
def log(s):print(f'[{time.time()-start:.1f}s] {s}',flush=True)
def coords(m,field='co'):
 a=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get(field,a);return a.reshape(-1,3)
fit=np.load(root/'work/inner_wall_fit.npz');seamfit=np.load(root/'work/seam_fit.npz')
def surface(v,sign):
 t=(.5*v[:,0]+.866025403784*v[:,2]-140)/60;d=(.866025403784*v[:,0]-.5*v[:,2]+90.8)/10
 return sum(c*t**i*d**j for c,(i,j) in zip(fit['coeff'][0 if sign<0 else 1],fit['powers']))
bpy.ops.wm.open_mainfile(filepath=str(root/'work/before_interior.blend'))
o=bpy.data.objects['DANYA-face-back'];m=o.data;v=coords(m);n=coords(m,'normal')
oldnorm=np.empty(len(m.loops)*3,dtype=np.float32);m.corner_normals.foreach_get('vector',oldnorm);oldnorm=oldnorm.reshape(-1,3)
oldloops=np.empty(len(m.loops),dtype=np.int32);m.loops.foreach_get('vertex_index',oldloops)
oldstart=np.array([p.loop_start for p in m.polygons]);oldtotal=np.array([p.loop_total for p in m.polygons])
dd=.866025403784*v[:,0]-.5*v[:,2]+90.8;sgn=np.where(v[:,1]<0,-1,1);pred=np.where(sgn<0,-surface(v,-1),surface(v,1));depth=abs(v[:,1])-pred
yz=(v[:,1:]-[0,135])/100;sx=sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(seamfit['coeff'],seamfit['powers']))
selected=(abs(dd)<4)&(v[:,2]>105)&(v[:,2]<216)&(depth<2.5)&((depth<-2.5)|(n[:,1]*sgn<.15))&(abs(v[:,0]-sx)>1)
bm=bmesh.new();bm.from_mesh(m);bm.verts.ensure_lookup_table();bm.faces.ensure_lookup_table()
vid=bm.verts.layers.int.new('interior_original_vertex');fid=bm.faces.layers.int.new('interior_original_face')
for i,x in enumerate(bm.verts):x[vid]=i+1
for i,f in enumerate(bm.faces):f[fid]=i+1
removed=int(sum(selected));bmesh.ops.delete(bm,geom=[bm.verts[i] for i in np.where(selected)[0]],context='VERTS');log(f'Removed {removed} panel/attachment vertices')
unseen=set(bm.verts);components=[]
while unseen:
 seed=unseen.pop();stack=[seed];component=[]
 while stack:
  cur=stack.pop();component.append(cur)
  for e in cur.link_edges:
   other=e.other_vert(cur)
   if other in unseen:unseen.remove(other);stack.append(other)
 components.append(component)
components.sort(key=len,reverse=True);log('Components after panel cut: '+str([len(c) for c in components]))
for component in components[1:]:
 assert len(component)<1000 and all(abs(.866025403784*x.co.x-.5*x.co.z+90.8)<6 for x in component),'Unexpected detached shell section'
 bmesh.ops.delete(bm,geom=component,context='VERTS')
boundary=[e for e in bm.edges if e.is_boundary]
# All openings must be confined to the two inner side-wall attachment patches.
seen=set();groups=[]
for e in boundary:
 if e in seen:continue
 stack=[e];group=[];seen.add(e)
 while stack:
  cur=stack.pop();group.append(cur)
  for x in cur.verts:
   for ee in x.link_edges:
    if ee.is_boundary and ee not in seen:seen.add(ee);stack.append(ee)
 groups.append(group)
log('Attachment boundary lengths: '+str([len(g) for g in groups]))
for group in groups:
 pts=np.array([tuple(x.co) for e in group for x in e.verts]);log('Patch bounds '+str([pts.min(0).tolist(),pts.max(0).tolist()]))
assert len(groups)==2,'Unexpected cut topology; do not publish'
assert all(all(x.co.y>20 for e in g for x in e.verts) or all(x.co.y<-20 for e in g for x in e.verts) for g in groups),'Cut reached outside side-wall attachments'
patch_boundary=set(x for e in boundary for x in e.verts)
patch_boundary_coords=np.array([tuple(x.co) for x in patch_boundary])
restore={(float(p[0]),float(p[2]),1 if p[1]>0 else -1):p.copy() for p in patch_boundary_coords}
for x in patch_boundary:x.co.y=60.0 if x.co.y>0 else -60.0
patch=bmesh.ops.holes_fill(bm,edges=boundary,sides=0)['faces']
for f in patch:f[fid]=0
bmesh.ops.triangulate(bm,faces=patch)
for iteration in range(5):
 newfaces={f for f in bm.faces if f[fid]==0}
 edges=[e for e in bm.edges if len(e.link_faces)==2 and all(f in newfaces for f in e.link_faces) and e.calc_length()>1.2]
 log(f'Patch refinement {iteration}: {len(edges)} long edges')
 if not edges:break
 bmesh.ops.subdivide_edges(bm,edges=edges,cuts=1,use_grid_fill=False)
 bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if f[fid]==0])
for x in bm.verts:
 key=(float(x.co.x),float(x.co.z),1 if x.co.y>0 else -1)
 if abs(abs(x.co.y)-60)<.00001 and key in restore:x.co=restore[key]
original_lookup={tuple(p):i+1 for i,p in enumerate(v)}
vid=bm.verts.layers.int['interior_original_vertex']
for x in bm.verts:x[vid]=original_lookup.get(tuple(x.co),0)
log('Preserved original vertices: '+str(sum(x[vid]>0 for x in bm.verts)))
for sign in [-1,1]:
 bc=patch_boundary_coords[patch_boundary_coords[:,1]*sign>0];res=bc[:,1]-surface(bc,sign)
 kd=KDTree(len(bc))
 for i,p in enumerate(bc):kd.insert((float(p[0]),0,float(p[2])),i)
 kd.balance()
 new=[x for x in bm.verts if x[vid]==0 and x.co.y*sign>0];nv=np.array([tuple(x.co) for x in new]);yy=surface(nv,sign)
 for j,x in enumerate(new):
  near=kd.find_n((x.co.x,0,x.co.z),6);weights=np.array([1/max(r[2],.05)**2 for r in near]);correction=sum(w*res[r[1]] for w,r in zip(weights,near))/weights.sum();x.co.y=float(yy[j]+correction)
log('Reconstructed curved inner side walls')
for f in bm.faces:
 if f[fid]==0:f.smooth=True
bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(m);bm.free();m.update()
# Preserve the existing custom normals exactly on untouched faces, including the exterior and joint.
vn=np.empty(len(m.vertices),dtype=np.int32);m.attributes['interior_original_vertex'].data.foreach_get('value',vn)
fn=np.empty(len(m.polygons),dtype=np.int32);m.attributes['interior_original_face'].data.foreach_get('value',fn)
normals=np.empty(len(m.loops)*3,dtype=np.float32);m.corner_normals.foreach_get('vector',normals);normals=normals.reshape(-1,3)
for p in m.polygons:
 fi=fn[p.index]-1
 if fi<0:continue
 a=oldstart[fi];b=a+oldtotal[fi];lookup={int(oldloops[k]):oldnorm[k] for k in range(a,b)}
 for li in p.loop_indices:
  orig=vn[m.loops[li].vertex_index]-1
  if orig in lookup:normals[li]=lookup[orig]
m.normals_split_custom_set(normals.tolist());m.update()
o['interior_cleanup']='Internal inclined panel removed; two side-wall attachment patches reconstructed. Exterior and mating interface preserved.'
m.calc_loop_triangles();faces=np.empty(len(m.loop_triangles)*3,dtype=np.int32);m.loop_triangles.foreach_get('vertices',faces)
np.savez(root/'work/interior_back.npz',v=coords(m),f=faces.reshape(-1,3),original_vertex=vn)
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/interior_candidate.blend'))
(root/'work/interior_process.json').write_text(json.dumps({'removed_vertices':removed,'attachment_boundary_edges':[len(g) for g in groups],'new_patch_faces':int(sum(fn==0))},indent=2));log('Saved candidate')
