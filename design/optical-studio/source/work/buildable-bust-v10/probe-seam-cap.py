import numpy as np,json
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
p=Path('outputs/danya-simulator/assets/torso-flat-study');v=np.fromfile(p/'face.positions.bin',np.float32).reshape(-1,3);f=np.fromfile(p/'face.indices.bin',np.uint32).reshape(-1,3);bv=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);s=np.load('work/buildable-bust-v10/seam.npy');fit=np.load('work/seam_fit.npz')
def fx(y,z):
 y=y/1.1/100;z=(z/1.1-135)/100;return 1.1*sum(c*y**i*z**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
rows=[]
for z in [138,162]:
 for sign in [-1,1]:
  sub=s[s[:,1]*sign>40];pt=sub[np.argmin(abs(sub[:,2]-z))];row={'z':float(pt[2]),'edge':pt.tolist(),'hits':[]}
  for off in [.1,.3,.5,.8,1.,1.5,2.,3.,4.]:
   y=float(pt[1]-sign*off);zz=float(pt[2]);hit=bv.ray_cast(Vector((-30,y,zz)),Vector((1,0,0)),150)[0];second=bv.ray_cast(hit+Vector((.001,0,0)),Vector((1,0,0)),150)[0] if hit is not None else None;row['hits'].append({'off':off,'expectedX':fx(y,zz),'hit':list(hit)if hit is not None else None,'second':list(second)if second is not None else None})
  rows.append(row)
Path('work/buildable-bust-v10/seam-cap.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
