exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'smooth-head-v4'
def direct(a,b,mode):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
head=loadpath(out/'head-upper.stl');glass=loadpath(src/'assembly/mirror-glass.stl');vs=np.array([v.co[:]for v in glass.data.vertices]);c=vs.mean(0);cov=(vs-c).T@(vs-c);_,axes=np.linalg.eigh(cov);local=(vs-c)@axes;extent=np.ptp(local,axis=0);local*=1+.5/extent;coords=c+local@axes.T
for v,p in zip(glass.data.vertices,coords):v.co=p
direct(head,glass,'DIFFERENCE')
carrier=loadpath(out/'magnet-carrier.stl');bpy.ops.mesh.primitive_cube_add(size=1,location=(41.6,-15.5,258));box=bpy.context.object;box.dimensions=(6,8,6);bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);direct(carrier,box,'INTERSECT');shift=carrier.copy();shift.data=carrier.data.copy();bpy.context.collection.objects.link(shift);shift.location.x=-.3;direct(carrier,shift,'UNION');direct(head,carrier,'DIFFERENCE')
for y in [-18.054,18.054]:
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=1.65,depth=3,location=(41.8,y,255),rotation=(0,math.pi/2,0));c=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);direct(head,c,'DIFFERENCE')
print(export(head,'head-upper'),flush=True)
