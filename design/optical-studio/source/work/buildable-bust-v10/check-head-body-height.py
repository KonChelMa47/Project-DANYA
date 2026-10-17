import json,numpy as np
from pathlib import Path
r=Path('outputs/danya-simulator/assets');cx=-22.01616859436035;cy=.3457450866699219
body=np.fromfile(r/'buildable-bust-v10/body-with-floor-bosses.positions.bin',np.float32).reshape(-1,3);top=float(body[:,2].max());rows=[]
def minimum(a,b,lo,hi):
 ans=np.minimum(a*np.sin(lo)+b*np.cos(lo),a*np.sin(hi)+b*np.cos(hi));phase=np.arctan2(a,b)+np.pi
 for k in [-2,-1,0,1,2]:
  t=phase+k*2*np.pi;ok=(t>=lo)&(t<=hi);ans=np.where(ok,np.minimum(ans,a*np.sin(t)+b*np.cos(t)),ans)
 return ans
for name,path in [('head','buildable-bust-v10/head-inner-relief'),('face','torso-flat-study/face')]:
 v=np.fromfile(r/(path+'.positions.bin'),np.float32).reshape(-1,3).astype(float);v[:,2]+=8;B=24+minimum(v[:,1]-cy,v[:,2]-28,np.deg2rad(-6),np.deg2rad(6));z=4+minimum(-(v[:,0]-cx),B,0,np.deg2rad(12));i=int(np.argmin(z));rows.append({'name':name,'minimumWorldZMM':float(z[i]),'bodyMaximumZMM':top,'guaranteedVerticalGapMM':float(z[i]-top-.0001),'criticalVertex':i})
report={'scope':'Exact mesh vertex minimum worldZ over continuous pitch0..12,roll±6. Yaw arbitrary (does not changeZ). All triangle points included by linearity at each pose. Current8mm common lift, pitchpivot4,rollpivot28.0.0001mm numeric allowance. Separation from fixed body only; neck cover, internal mechanisms and deformation excluded.','rows':rows};(r/'buildable-bust-v10/continuous-head-body-height.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
