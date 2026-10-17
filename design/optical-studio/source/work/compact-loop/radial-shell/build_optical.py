import numpy as np,json
from split_fans import split_fans
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
v=np.fromfile(root/'work/adult-kit/original-head.positions.bin','<f4').reshape(-1,3);f=np.fromfile(root/'work/adult-kit/head-outer-original.indices.bin','<u4').reshape(-1,3);used=np.unique(f);remap=np.zeros(len(v),dtype=np.uint32);remap[used]=np.arange(len(used));v=v[used].astype(float);f=remap[f];v,f,source=split_fans(v,f);n=np.zeros_like(v);cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
for j in range(3):np.add.at(n,f[:,j],cross)
n/=np.linalg.norm(n,axis=1)[:,None];edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);_,first,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True);boundary=edges[first[count==1]];N=len(v)
for thickness in [.6,.8,1.0]:
 for cx in [-4]:
  center=np.array([cx,0,205]);d=center-v;d/=np.linalg.norm(d,axis=1)[:,None];d=np.fromfile(p/'optical-directions.bin','<f4').reshape(-1,3).astype(float)[source];cosine=-(n*d).sum(1)
  bad=cosine<=.02;fallback=center-v;fallback/=np.linalg.norm(fallback,axis=1)[:,None];d[bad]=fallback[bad];cosine=-(n*d).sum(1)
  travel=np.minimum(thickness/cosine,10);inner=v+d*travel[:,None];vv=np.concatenate([v,inner]);ff=np.concatenate([f,f[:,::-1]+N,np.array([[b,a,a+N] for a,b in boundary]),np.array([[b,a+N,b+N] for a,b in boundary])]);name=f'optical-t{thickness}'
  vv.astype('<f4').tofile(p/f'{name}.positions.bin');ff.astype('<u4').tofile(p/f'{name}.indices.bin');(p/f'{name}.json').write_text(json.dumps({'center':center.tolist(),'normalThicknessTarget':thickness,'minNormalVertexDisplacement':float((travel*cosine).min()),'outerVertices':N,'triangles':len(ff),'outerPreservedExactlyFloat32':bool(np.array_equal(vv[:N].astype(np.float32),v.astype(np.float32)))}));print(name,len(vv),len(ff))
