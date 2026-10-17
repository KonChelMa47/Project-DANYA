from pathlib import Path
import numpy as np, json, hashlib, struct
root=Path(__file__).resolve().parent.parent
out=root/'outputs/danya-simulator/assets'
manifest={'units':'mm','unit_assumption':'1 STL coordinate unit = 1 mm','source':'robot_shell_clean_interior.blend','meshes':{}}
for name,src in [('head','interior_back.npz'),('face','DANYA-face-shell-repaired.npz')]:
 d=np.load(root/'work'/src); v=d['v'].astype('<f4'); f=d['f'].astype('<u4')
 tri=v[f]; cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); norm=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-12)
 vn=np.zeros_like(v)
 for k in range(3): np.add.at(vn,f[:,k],cross)
 vn/=np.maximum(np.linalg.norm(vn,axis=1)[:,None],1e-12)
 for suffix,arr in [('positions',v),('indices',f),('normals',vn)]: (out/f'{name}.{suffix}.bin').write_bytes(arr.tobytes())
 stl=np.zeros(len(f),dtype=[('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]);stl['normal']=norm;stl['vertices']=tri
 (out/f'DANYA-{name}.stl').write_bytes(b'Project DANYA cleaned shell; coordinate unit assumed mm'.ljust(80,b' ')+struct.pack('<I',len(f))+stl.tobytes())
 manifest['meshes'][name]={'vertices':len(v),'triangles':len(f),'min':v.min(axis=0).tolist(),'max':v.max(axis=0).tolist(),'sha256':hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
