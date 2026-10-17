import numpy as np,json,struct,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];out=root/'outputs/danya-simulator/assets/compact-optical-v2';out.mkdir(exist_ok=True)
manifest={'status':'EXPERIMENTAL - not an actual-projector validation or manufacturing release','exteriorScale':1.1,'headInterior':'Nominal 1 mm normal offset along optical rays; up to 10 mm offset travel; 0.6 mm internal seam bevel','source':'work/compact-loop/margin-candidate.json'}
for name,vpath,fpath in [('head',p/'bevel-0.6.positions.bin',p/'bevel-0.6.indices.bin'),('face',root/'work/adult-kit/original-face.positions.bin',root/'outputs/danya-simulator/assets/face.indices.bin')]:
 v=np.fromfile(vpath,'<f4').reshape(-1,3);f=np.fromfile(fpath,'<u4').reshape(-1,3);tri=v[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);smooth=np.zeros_like(v)
 for j in range(3):np.add.at(smooth,f[:,j],norm)
 smooth/=np.maximum(np.linalg.norm(smooth,axis=1)[:,None],1e-20);v.tofile(out/f'{name}.positions.bin');f.tofile(out/f'{name}.indices.bin');smooth.astype('<f4').tofile(out/f'{name}.normals.bin')
 norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-20);records=np.zeros(len(f),dtype=[('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')]);records['normal']=norm;records['vertices']=tri
 with open(out/f'DANYA-{name}-full-coverage.stl','wb') as fp:fp.write(b'DANYA experimental compact optical v2 - unverified'.ljust(80,b'\0'));fp.write(struct.pack('<I',len(f)));fp.write(records.tobytes())
 manifest[name]={'vertices':len(v),'triangles':len(f),'bounds':[v.min(0).tolist(),v.max(0).tolist()]}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));s=json.loads((root/'work/compact-loop/margin-candidate.json').read_text());s['design']='compact-optical-v2';(out/'layout.json').write_text(json.dumps(s,indent=2));shutil.copy2(p/'nominal-mask.bin',out/'ghost-mask.bin');print(out)
