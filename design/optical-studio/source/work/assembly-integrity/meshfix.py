import sys,json,time
from pathlib import Path
import numpy as np
import pymeshfix
r=Path('outputs/danya-simulator/assets/projection-prototype/solid-seam-v2');name=sys.argv[1];dst=sys.argv[2] if len(sys.argv)>2 else name+'-meshfix'
raw=np.fromfile(r/(name+'.stl'),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'];v,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3)
def topology(v,f):
 e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);u,c=np.unique(e,axis=0,return_counts=True)
 return {'vertices':len(v),'triangles':len(f),'nonTwoFaceEdges':int((c!=2).sum()),'boundaryEdges':int((c==1).sum()),'volumeMM3':float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)}
rep={'source':name,'pymeshfixVersion':pymeshfix.__version__,'before':topology(v,f)};print(rep,flush=True);t=time.time();m=pymeshfix.PyTMesh(False);m.load_array(v.astype(float),f.astype(np.int32));ok=m.clean(max_iters=10,inner_loops=3);w,g=m.return_arrays();rep['cleanReturn']=str(ok);rep['after']=topology(w,g);rep['elapsedSeconds']=time.time()-t
tri=w[g];out=np.zeros(len(g),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);out['v']=tri;n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30);out['n']=n
with open(r/(dst+'.stl'),'wb')as h:h.write(b'DANYA MeshFix local repair'.ljust(80,b'\0'));h.write(np.asarray([len(g)],dtype='<u4').tobytes());h.write(out.tobytes())
(r/(dst+'-report.json')).write_text(json.dumps(rep,indent=2));print(rep,flush=True)
