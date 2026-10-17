exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'unified-head-v5'
def boolean(a,b,mode):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def cyl(c,r,depth):
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=r,depth=depth,location=c,rotation=(0,math.pi/2,0));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
def rib(x0,x1,y,z,sign):
 dy=sign*12;dz=-8;ll=(dy*dy+dz*dz)**.5;ny=-dz/ll*2.5;nz=dy/ll*2.5;p=[(y+ny,z+nz),(y-ny,z-nz),(y+dy-ny,z+dz-nz),(y+dy+ny,z+dz+nz)];v=[(x,yy,zz)for x in [x0,x1]for yy,zz in p];f=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)];m=bpy.data.meshes.new('rib');m.from_pydata(v,[],f);m.update();o=bpy.data.objects.new('rib',m);bpy.context.collection.objects.link(o);clean(o);return o
for name,side in [('magnet-carrier','face')]:
 o=loadpath(out/'magnet-carrier-localized-clean.stl')
 for i in [2,3]:
  m=loadpath(src/'assembly'/f'magnet-clear-{i}-{side}-magnet.stl');v=np.array([p.co[:]for p in m.data.vertices]);mn=v.min(0);mx=v.max(0);c=(mn+mx)/2;bpy.data.objects.remove(m,do_unlink=True);x0=mn[0]-(1 if side=='head'else .05);x1=mx[0]+(.05 if side=='head'else 1)
  boss=cyl(((x0+x1)/2,c[1],c[2]),4.4,x1-x0);a=Vector((x1-.5,c[1],c[2]-3));b=Vector((34,c[1],c[2]-6));bpy.ops.mesh.primitive_cube_add(size=1,location=(a+b)/2);beam=bpy.context.object;beam.dimensions=(3,5,(b-a).length+.3);beam.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler();bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);boolean(boss,beam,'UNION')
  # Keep additions inside the smooth outside envelope.
  boolean(boss,loadpath(out/('outer-envelope.stl'if side=='head'else 'carrier-inset-envelope.stl')),'INTERSECT')
  boolean(o,boss,'UNION');start=mn[0]-.1 if side=='head'else mn[0]-10;end=mx[0]+10 if side=='head'else mx[0]+.1
  boolean(o,cyl(((start+end)/2,c[1],c[2]),3.35,end-start),'DIFFERENCE')
 print(name,export(o,name+'-magnet-rebuilt'),flush=True)
