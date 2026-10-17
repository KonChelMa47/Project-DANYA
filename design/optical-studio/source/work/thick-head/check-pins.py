import numpy as np,json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');rows=[]
for i in range(2):
 x=[]
 for folder in ['assembly','head-split-quarter']:
  a=np.fromfile(r/folder/f'head-alignment-pin-{i}.stl',dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'];x.append((a.max((0,1))-a.min((0,1))).tolist())
 rows.append({'pin':i,'centralPinDimensionsMM':x[0],'quarterPinDimensionsMM':x[1],'sameDimensionsWithin01MM':bool(np.allclose(x[0],x[1],atol=.01))})
(r/'thick-head-v3/pin-reuse.json').write_text(json.dumps(rows,indent=2));print(rows)
