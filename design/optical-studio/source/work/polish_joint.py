import bpy,numpy as np,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent
bpy.ops.wm.open_mainfile(filepath=str(root/'work/common_simplified.blend'))
o=next(o for o in bpy.context.scene.objects if o.type=='MESH');bpy.context.view_layer.objects.active=o
fit=np.load(root/'work/seam_fit.npz');v=np.empty(len(o.data.vertices)*3,dtype=np.float32);o.data.vertices.foreach_get('co',v);v=v.reshape(-1,3)
yz=(v[:,1:]-[0,135])/100;fx=sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
w=np.exp(-((v[:,0]-fx)/6)**4);g=o.vertex_groups.new(name='Final joint polish')
for k in range(1,33):
 ids=np.where((w>(k-1)/32)&(w<=k/32))[0]
 if len(ids):g.add(ids.tolist(),k/32,'REPLACE')
m=o.modifiers.new('Remove residual seam ridge','SMOOTH');m.factor=.5;m.iterations=240;m.vertex_group=g.name
bpy.ops.object.modifier_apply(modifier=m.name)
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/common_simplified.blend'))
print('Polished original joint on common surface',flush=True)
