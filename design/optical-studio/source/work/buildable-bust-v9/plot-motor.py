import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path
p=Path('outputs/danya-simulator/assets/buildable-bust-v2/st3215-official.stl');b=p.read_bytes();n=int.from_bytes(b[80:84],'little');a=np.frombuffer(b, dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),offset=84,count=n)['v'].copy();a += [25.5,-9.6,0]
f,axs=plt.subplots(1,3,figsize=(16,6))
for ax,(x,y,z) in zip(axs,[(0,1,2),(0,2,1),(2,1,0)]):
 i=np.argsort(a[:,:,z].mean(axis=1));tri=a[i][:,:,[x,y]];depth=a[i,:,z].mean(axis=1);coll=PolyCollection(tri,facecolors=plt.cm.gray((depth-depth.min())/(depth.max()-depth.min())*.7+.15),edgecolors='none');ax.add_collection(coll);ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('XYZ'[x]);ax.set_ylabel('XYZ'[y]);ax.grid(alpha=.2)
f.tight_layout();f.savefig('work/buildable-bust-v9/motor-views.png',dpi=140)
