from pathlib import Path
import numpy as np,json,hashlib,re,zipfile,xml.etree.ElementTree as ET
r=Path('outputs/danya-simulator/assets/projection-prototype/printable-seam');out={};dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
for name in ['head-upper','head-lower','magnet-carrier']:
 p=r/(name+'.stl');raw=np.fromfile(p,dtype=dt,offset=84)['v'];v,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3);e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);u,c=np.unique(e,axis=0,return_counts=True);zero=int((np.linalg.norm(np.cross(raw[:,1].astype(float)-raw[:,0],raw[:,2].astype(float)-raw[:,0]),axis=1)<1e-10).sum());assert np.all(c==2)and zero==0,(name,zero,int((c!=2).sum()));assert p.read_bytes()==(r/'print'/(name+'.stl')).read_bytes()
 with zipfile.ZipFile(r/'print'/(name+'.3mf'))as z:
  root=ET.fromstring(z.read('3D/3dmodel.model'));verts=root.findall('.//{*}vertex');w=np.array([[float(p.get(k))for k in ['x','y','z']]for p in verts]);assert np.array_equal(v,w)
 mins=np.full(3,np.inf);maxs=-mins;pos=np.zeros(3);feat="Custom"
 for line in (Path('work/printable-seam/slices')/name/'plate_1.gcode').open():
  if line.startswith('; FEATURE:'):feat=line.split(':',1)[1].strip()
  if not line.startswith(('G0 ','G1 ')):continue
  vals={k:float(v)for k,v in re.findall(r'([XYZE])(-?[\d.]+)',line.split(';')[0])};q=np.array([vals.get(k,pos[i])for i,k in enumerate('XYZ')]);
  if feat!='Custom' and vals.get('E',0)>0 and np.linalg.norm(q[:2]-pos[:2])>0:mins=np.minimum(mins,np.minimum(pos,q));maxs=np.maximum(maxs,np.maximum(pos,q))
  pos=q
 out[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'triangles':len(f),'nonManifoldEdges':int((c!=2).sum()),'zeroAreaTriangles':zero,'threeMFMeshMatchesSTL':True,'extrusionBoundsMM':[mins.tolist(),maxs.tolist()],'insideAD5MIncluding025MMHalfWidth':bool(np.all(mins[:2]-.25>=-110)and np.all(maxs[:2]+.25<=110)and maxs[2]<=220),'surfaceCheck':json.loads((r/(name+'-surface.json')).read_text())}
(r/'FINAL-VALIDATION.json').write_text(json.dumps(out,indent=2));print({n:{k:v for k,v in d.items()if k!='surfaceCheck'}for n,d in out.items()})
