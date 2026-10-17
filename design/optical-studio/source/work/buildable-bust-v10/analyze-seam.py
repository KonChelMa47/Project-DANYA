import numpy as np,json
from pathlib import Path
from mathutils.kdtree import KDTree
p=Path('outputs/danya-simulator/assets/torso-flat-study')
a=np.fromfile(p/'head.positions.bin',np.float32).reshape(-1,3);b=np.fromfile(p/'face.positions.bin',np.float32).reshape(-1,3)
print('head',a.shape,a.min(0),a.max(0));print('face',b.shape,b.min(0),b.max(0))
k=KDTree(len(a))
for i,v in enumerate(a):k.insert(v,i)
k.balance();s=[]
for v in b:
 c,i,d=k.find(v)
 if d<.08:s.append((a[i]+v)/2)
s=np.array(s);print('joint',len(s),s.min(0),s.max(0));np.save('work/buildable-bust-v10/seam.npy',s)
for z in [60,90,120,150,180,210,240,270]:
 t=s[abs(s[:,2]-z)<2]
 if len(t):print(z,t.min(0),t.max(0))
