import numpy as np,json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');d=np.load(r/'face-v8/screen-data.npz');raw=np.fromfile(r/'unified-head-v5/face.stl',dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v']
def keys(t):
 t=np.asarray(t,dtype='<f4');p=np.sort(t.reshape(-1,3).copy().view('V12').reshape(-1,3),axis=1);return p.copy().view('V36').reshape(-1)
a=keys(d['after'][d['innerFaces']]);b=keys(raw);found=np.isin(a,b);p=r/'unified-head-v5/face-perimeter-change.json';j=json.loads(p.read_text());j.update(innerTrianglesChecked=len(a),innerTrianglesExactlyPreserved=int(found.sum()),innerPreservationCheck='Exact float32 triangle coordinates, winding independent.');p.write_text(json.dumps(j,indent=2));print(j);assert found.all()
