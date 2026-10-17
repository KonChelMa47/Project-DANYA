import bpy,numpy as np,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
bpy.ops.wm.open_mainfile(filepath=str(root/'work/before_interior.blend'))
m=bpy.data.objects['DANYA-face-back'].data
v=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get('co',v);v=v.reshape(-1,3)
n=np.empty_like(v);m.vertices.foreach_get('normal',n.ravel())
dd=.866025403784*v[:,0]-.5*v[:,2]+90.8;tt=.5*v[:,0]+.866025403784*v[:,2]
print('panel slab vertices',int(sum((abs(dd)<2)&(v[:,2]>105)&(v[:,2]<215))),flush=True)
power=np.array([(i,j) for i in range(5) for j in range(3) if i+j<=4])
coefs=[];reports=[]
for sign in [-1,1]:
 sel=(v[:,1]*sign>30)&(n[:,1]*sign<-.35)&(abs(dd)>4)&(abs(dd)<13)&(v[:,2]>98)&(v[:,2]<220)
 t=(tt[sel]-140)/60;d=dd[sel]/10;A=np.array([t**i*d**j for i,j in power]).T;y=v[sel,1]
 w=np.ones(len(y))
 for _ in range(6):
  c=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0];res=A@c-y;w=1/np.maximum(1,abs(res)/.25)
 coefs.append(c);reports.append({'side':sign,'samples':len(y),'residual':np.percentile(abs(res),[50,90,99,100]).tolist()})
np.savez(root/'work/inner_wall_fit.npz',coeff=np.array(coefs),powers=power)
print(json.dumps(reports),flush=True)
