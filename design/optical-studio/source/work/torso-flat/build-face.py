import numpy as np,json,sys
from pathlib import Path
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p/'shell'))
from split_fans import split_fans
v=np.fromfile(p.parent/'adult-kit/original-face.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p.parent/'adult-kit/original-face-outer.indices.bin','<u4').reshape(-1,3);used=np.unique(f);remap=np.zeros(len(v),dtype=np.uint32);remap[used]=np.arange(len(used));v=v[used].astype(float);f=remap[f];v,f,source=split_fans(v,f);N=len(v);n=np.zeros_like(v);cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
for j in range(3):np.add.at(n,f[:,j],cross)
n/=np.linalg.norm(n,axis=1)[:,None];d=np.array(json.load(open(p/'virtual-source.json')))-v;d/=np.linalg.norm(d,axis=1)[:,None];cos=-(n*d).sum(1);print('cos',np.percentile(cos,[0,1,50,100]));assert cos.min()>0
edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);_,first,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True);boundary=edges[first[count==1]]
# Fixed radial fractional contraction preserves correspondence and avoids variable-offset folds.
center=np.array(json.load(open(p/'virtual-source.json')))
for fraction in [.003,.005,0.8]:
 inner=v+(center-v)*fraction if fraction<.1 else v+d*(fraction/cos)[:,None];vv=np.concatenate([v,inner]);ff=np.concatenate([f,f[:,::-1]+N,np.array([[b,a,a+N] for a,b in boundary]),np.array([[b,a+N,b+N] for a,b in boundary])]);name=f'face-radial-{fraction}';vv.astype('<f4').tofile(p/(name+'.positions.bin'));ff.astype('<u4').tofile(p/(name+'.indices.bin'));f.astype('<u4').tofile(p/(name+'.outer.indices.bin'));thickness=np.linalg.norm(inner-v,axis=1)*cos;report={'fraction':fraction,'outerVertices':N,'outerTriangles':len(f),'normalVertexSeparationMM':np.percentile(thickness,[0,50,100]).tolist(),'scope':'Vertex normal projected separation; not minimum wall thickness or fabrication validation'};(p/(name+'.json')).write_text(json.dumps(report,indent=2));print(report)
