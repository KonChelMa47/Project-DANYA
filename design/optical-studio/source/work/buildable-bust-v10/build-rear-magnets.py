import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
root=Path.cwd();out=root/'outputs/danya-simulator/assets/buildable-bust-v10';base=root/'outputs/danya-simulator/assets/torso-flat-study';fit=np.load(root/'work/seam_fit.npz')
def seam(y,z):
 y=y/1.1/100;z=(z/1.1-135)/100;return 1.1*sum(c*y**i*z**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
def mesh(name,v,f):
 m=bpy.data.meshes.new(name);m.from_pydata(v,[],f);m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(m);bm.free();return o
def boolean(a,b,op):
 bpy.context.view_layer.objects.active=a;mod=a.modifiers.new(op,'BOOLEAN');mod.operation=op;mod.solver='EXACT';mod.use_self=True;mod.object=b;bpy.ops.object.modifier_apply(modifier=mod.name)
def cylinder(name,center,r,length):
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=r,depth=length,location=center,rotation=(0,math.pi/2,0));o=bpy.context.object;o.name=name;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
def report(o):
 bm=bmesh.new();bm.from_mesh(o.data);r={'vertices':len(bm.verts),'faces':len(bm.faces),'boundaryEdges':sum(e.is_boundary for e in bm.edges),'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'volume':bm.calc_volume(signed=False)};bm.free();return r
def export(o):
 bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(out/(o.name+'.stl')),export_selected_objects=True);o.data.calc_loop_triangles();np.array([v.co[:]for v in o.data.vertices],np.float32).tofile(out/(o.name+'.positions.bin'));np.array([t.vertices[:]for t in o.data.loop_triangles],np.uint32).tofile(out/(o.name+'.indices.bin'));return report(o)
def grid_solid(name,grid,nx,ny):
 faces=[];N=nx*ny
 for layer in [0,1]:
  for i in range(nx-1):
   for j in range(ny-1):
    a=layer*N+i*ny+j;faces.append((a,a+1,a+ny+1,a+ny))
 edge=list(range(ny))+[i*ny+ny-1 for i in range(1,nx)]+[(nx-1)*ny+j for j in range(ny-2,-1,-1)]+[i*ny for i in range(nx-2,0,-1)]
 for a,b in zip(edge,edge[1:]+edge[:1]):faces.append((a,b,b+N,a+N))
 return mesh(name,grid,faces)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
v=np.fromfile(base/'head.positions.bin',np.float32).reshape(-1,3);f=np.fromfile(base/'head.indices.bin',np.uint32).reshape(-1,3);wall=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);head=mesh('head-magnetic-rebates',v.tolist(),f.tolist());initial=report(head)
facev=np.fromfile(base/'face.positions.bin',np.float32).reshape(-1,3);facef=np.fromfile(base/'face.indices.bin',np.uint32).reshape(-1,3);faceWall=BVHTree.FromPolygons(facev.tolist(),facef.tolist(),all_triangles=True)
seams=np.load(root/'work/buildable-bust-v10/seam.npy');rows=[]
selected=[r for r in json.load(open(root/'work/buildable-bust-v10/rear-magnet-four-search.json'))if r['dx']==28 and not r['hits'] and not r['lightHits']]
for index,row in enumerate(selected):
 x,y,z=row['center'];sign=1 if y>0 else -1;sub=seams[seams[:,1]*sign>40];edge=sub[np.argmin(abs(sub[:,2]-z))];z=float(edge[2]);c=Vector((x,y,z));tag='magnet-rear-'+str(index)
 # Original face seam cap supplies a bonding surface. It is not offset forward.
 # A shallow local rebate in the head clears the return tongue; outer0.8mm rim retained.
 def flange(name,u0,u1,offset0,offset1,height):
  vv=[];n=9
  for u in [u0,u1]:
   for i in range(n):
    yy=float(edge[1])-sign*(offset0+(offset1-offset0)*i/(n-1))
    for j in range(n):
     zz=z-height/2+height*j/(n-1);vv.append((seam(yy,zz)+u,yy,zz))
  return grid_solid(name,vv,n,n)
 slot=flange(tag+'-head-rebate-tool',-1.65,.15,.8,5.2,12.4);export(slot);slot.hide_render=True;slot.hide_set(True)
 front=flange(tag+'-face-return',-1.4,-.2,1.,5.,12)
 # Smooth ribbon inside the rear shell joins the cap to the forward magnet cup.
 n=17;w=9;vv=[]
 for layer in [0,1]:
  for i in range(n):
   t=i/(n-1);xx=x+2.8+(float(edge[0])-1.-(x+2.8))*t
   for j in range(w):
    zz=z-4+8*j/(w-1);hit=wall.ray_cast(Vector((xx,0,zz)),Vector((0,sign,0)),100)[0]
    if hit is None:raise RuntimeError('missing wall ribbon')
    endY=float(edge[1])-sign*3.;yy=(1-t)*y+t*endY
    # Keep ribbon inside shell until its end enters the intentional rebate.
    if t<.85:yy=sign*min(abs(yy),abs(hit.y)-1.1)
    yy+=sign*(.8 if layer else -.8);vv.append((xx,yy,zz))
 ribbon=grid_solid(tag+'-ribbon',vv,n,w);boolean(front,ribbon,'UNION');bpy.data.objects.remove(ribbon,do_unlink=True)
 cup=cylinder(tag+'-face-cup',c+Vector((2.3,0,0)),4.2,4.2);boolean(front,cup,'UNION');bpy.data.objects.remove(cup,do_unlink=True)
 # Forward cup startsX+0.2; rear cup endsX0.0, preserving0.2mm carrier gap.
 bore=cylinder('front-bore',c+Vector((1.75,0,0)),3.075,3.3);boolean(front,bore,'DIFFERENCE');bpy.data.objects.remove(bore,do_unlink=True)
 back=cylinder(tag+'-head-cup',c+Vector((-2.1,0,0)),4.2,4.2)
 # A contoured foot reaches the head inner surface through a0.2mm bondline.
 vv=[];n=9
 for layer in [0,1]:
  for i in range(n):
   xx=x-5.5+7*i/(n-1)
   for j in range(n):
    zz=z-6+12*j/(n-1);hit=wall.ray_cast(Vector((xx,0,zz)),Vector((0,sign,0)),100)[0]
    yy=hit.y-sign*.2 if layer else y+sign*2.5;vv.append((xx,yy,zz))
 foot=grid_solid(tag+'-head-foot',vv,n,n);boolean(back,foot,'UNION');bpy.data.objects.remove(foot,do_unlink=True)
 bore=cylinder('rear-bore',c+Vector((-1.6,0,0)),3.075,3.4);boolean(back,bore,'DIFFERENCE');bpy.data.objects.remove(bore,do_unlink=True)
 rr=[]
 for o in [front,back]:rr.append({'name':o.name,'mesh':export(o)})
 for key,dx in [('face',1.8),('head',-1.6)]:
  mag=cylinder(tag+'-'+key+'-magnet',c+Vector((dx,0,0)),2.95,2.9);rr.append({'name':mag.name,'mesh':export(mag)})
 rows.append({'id':index,'center':list(c),'axis':[1,0,0],'seamEdge':edge.tolist(),'magnetDiameterMM':5.9,'magnetThicknessMM':2.9,'pocketDiameterMM':6.15,'carrierAxialGapMM':.2,'magnetFaceGapMM':.5,'parts':rr});print('BUILT',index,report(head),flush=True)
hr=report(head);ok=False # Head rebates will be evaluated by the dedicated solid Boolean pipeline.
if ok:export(head)
(out/'rear-magnet-mounts.json').write_text(json.dumps({'status':'candidate, geometric/optical/retention verification pending','headRebateValid':ok,'originalHead':initial,'rebatedHead':hr,'pairs':rows,'shellSeamOffsetMM':0,'notes':'Face seam kept at original coordinates. Four shallow head-only seam rebates clear face return tongues; original outer rim retained. Magnets pull face rearward alongX. Adhesive fit and pull-off strength not tested.'},indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(out/'rear-magnets-candidate.blend'))
print('HEAD_VALID',ok,hr,flush=True)
