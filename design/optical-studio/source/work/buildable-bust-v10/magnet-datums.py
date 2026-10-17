exec(open('work/buildable-bust-v10/analyze-seam.py').read().split("k=KDTree")[0])
from mathutils import Vector
from mathutils.bvhtree import BVHTree
surfaces={}
for name in ['head','face']:
 v=np.fromfile(p/(name+'.positions.bin'),np.float32).reshape(-1,3);f=np.fromfile(p/(name+'.indices.bin'),np.uint32).reshape(-1,3)
 surfaces[name]=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
s=np.load('work/buildable-bust-v10/seam.npy');rows=[]
for z in [90,150,210]:
 for sign in [-1,1]:
  subset=s[(np.abs(s[:,2]-z)<8)&(s[:,1]*sign>20)];pt=subset[np.argmin(abs(subset[:,2]-z))]
  # Fit local seam plane; outward direction in that plane locates inner pads.
  close=s[np.linalg.norm(s-pt,axis=1)<12];_,_,vv=np.linalg.svd(close-close.mean(0));n=vv[-1]
  if n[0]<0:n=-n
  # Seam points lie mostly on a curve, so plane normal uses original common surface polynomial gradient if underdetermined.
  n=np.array([1.,0.,0.])
  inner=pt.copy();inner[1]-=sign*5
  row={'id':f'{z}-{sign}','seam':pt.tolist(),'center':inner.tolist(),'axis':n.tolist(),'rays':{}}
  for key,bvh in surfaces.items():
   row['rays'][key]=[]
   for dx in [-5,-2,0,2,5]:
    org=Vector((float(inner[0]+dx),0,float(inner[2])));hit=bvh.ray_cast(org,Vector((0,sign,0)),100)
    row['rays'][key].append([dx,list(hit[0]) if hit[0] is not None else None])
  rows.append(row)
Path('work/buildable-bust-v10/magnet-datums.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
