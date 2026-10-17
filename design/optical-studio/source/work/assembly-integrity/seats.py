exec(open('work/assembly-integrity/rebuild.py').read().split("p=np.load(src/")[0])
import math
# Complete lower magnet cups and widen their roots into the existing edge, keeping original mating coordinates.
def cyl(c,radius,depth):
 bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=radius,depth=depth,location=c,rotation=(0,math.pi/2,0));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
bpy.ops.wm.stl_import(filepath=str(out/'head-lower.stl'));head=bpy.context.object
for x,y,z in [(19.351207733,-30.09,65.708442688),(19.533584595,30.09,64.766807556)]:
 front=x+1.45+.15;back=x-1.45-1.3;cup=cyl(((front+back)/2,y,z),4.4,front-back);op(head,cup,'UNION')
 # A broad radial root grows towards the side wall; no long unsupported bridge across the neck.
 sign=1 if y>0 else -1;root=box((x-1.5,y+sign*4,z),(3.0,9.0,7.0));op(head,root,'UNION')
 bore=cyl((x+.7,y,z),3.2,4.6);op(head,bore,'DIFFERENCE')
print(export(head,'head-lower'),flush=True)
