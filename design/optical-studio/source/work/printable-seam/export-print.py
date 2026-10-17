import numpy as np,json,sys
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'printable-seam';(out/'print').mkdir(exist_ok=True);dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);rows=json.loads((out/'print-transforms.json').read_text())if(out/'print-transforms.json').exists()else{};parts=json.loads((r/'parts.json').read_text())['parts']
for n in (sys.argv[1:] or ['head-upper','head-lower','magnet-carrier','carrier-before']):
 p=r/'assembly/magnet-carrier.stl'if n=='carrier-before'else out/(n+'.stl');d=np.fromfile(p,dtype=dt,offset=84);raw=d['v'].astype(float)
 if n.startswith('head-'):R=np.eye(3)if n=='head-upper'else np.diag([1,-1,-1])
 else:R=np.array([[0,2**-.5,-2**-.5],[0,2**-.5,2**-.5],[1,0,0]])
 q=raw@R.T;lo=q.min((0,1));hi=q.max((0,1));t=np.array([-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]]);q+=t;d['v']=q;d['n']=d['n']@R.T;path=out/'print'/(n+'.stl');path.write_bytes(b'DANYA printable seam millimeters'.ljust(80,b' ')+np.uint32(len(d)).tobytes()+d.tobytes());rows[n]={'R':R.tolist(),'t':t.tolist(),'sizeMM':(hi-lo).tolist()}
(out/'print-transforms.json').write_text(json.dumps(rows,indent=2));print(rows)
