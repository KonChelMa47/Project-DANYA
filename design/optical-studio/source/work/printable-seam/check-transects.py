import json,re,numpy as np
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype/printable-seam');meta=json.loads((r/'print-transforms.json').read_text());d=np.load(r/'band-samples.npz');head=np.load(r/'head-band-samples.npz');p=np.load(r.parent/'face-v12/perimeter-paths.npz');paths=p['paths'];dirs=p['dirs'];th=np.linspace(.25,.85,9)[:,None,None];sources={'carrier-before':(paths-dirs*th,paths),'magnet-carrier':(d['carrierInside'],d['carrierOutside']),'head-upper':(head['inside'],head['outside']),'head-lower':(head['inside'],head['outside'])};reports={}
def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
def ptseg(p,a,d):
 t=np.clip(np.sum((p-a)*d,axis=-1)/np.maximum(np.sum(d*d,axis=-1),1e-20),0,1);return np.linalg.norm(p-(a+t[...,None]*d),axis=-1)
for name,(a,b)in sources.items():
 stations=[1,3]if name.startswith('head-')else[2,4,6];a=a[stations,::8].reshape(-1,3);b=b[stations,::8].reshape(-1,3)
 if name.startswith('head-'):
  ok=a[:,2]>107.1 if name=='head-upper'else a[:,2]<106.3;a=a[ok];b=b[ok]
 R=np.array(meta[name]['R']);t=np.array(meta[name]['t']);a=a@R.T+t;b=b@R.T+t;layers={};pos=np.zeros(3);feat='Custom'
 for line in (Path('work/printable-seam/slices')/name/'plate_1.gcode').open():
  if line.startswith('; FEATURE:'):feat=line.split(':',1)[1].strip()
  if not line.startswith(('G0 ','G1 ')):continue
  vals={k:float(v)for k,v in re.findall(r'([XYZE])(-?[\d.]+)',line.split(';')[0])};q=np.array([vals.get(k,pos[i])for i,k in enumerate('XYZ')])
  if vals.get('E',0)>0 and feat not in ['Custom','Skirt','Brim','Wipe tower'] and not feat.startswith('Support')and np.linalg.norm(q[:2]-pos[:2])>0:layers.setdefault(round(q[2],3),[]).append([*pos,*q])
  pos=q
 layers={z:np.array(v)for z,v in layers.items()};zs=np.array(list(layers));ds=[]
 for start,end in zip(a,b):
  z=(start[2]+end[2])/2;near=zs[(zs>=min(start[2],end[2])-.13)&(zs<=max(start[2],end[2])+.13)];seg=np.concatenate([layers[z]for z in near])if len(near)else np.zeros((0,6))
  if not len(seg):ds.append(999);continue
  p0=start;v=end-p0;s=seg[:,:3];w=seg[:,3:]-s;aa=np.dot(v,v);bb=np.sum(v*w,axis=1);cc=np.sum(w*w,axis=1);delta=p0-s;dd=np.sum(v*delta,axis=1);ee=np.sum(w*delta,axis=1);den=aa*cc-bb*bb;valid=np.abs(den)>1e-15;tt=np.zeros(len(den));uu=tt.copy();tt[valid]=(bb*ee-cc*dd)[valid]/den[valid];uu[valid]=(aa*ee-bb*dd)[valid]/den[valid];ok=valid&(tt>=0)&(tt<=1)&(uu>=0)&(uu<=1);mid=np.linalg.norm(p0+tt[:,None]*v-s-uu[:,None]*w,axis=1);mid[~ok]=1e9
  dist=np.minimum.reduce([ptseg(p0,s,w),ptseg(end,s,w),ptseg(s,p0,v),ptseg(s+w,p0,v),mid]);ds.append(float(dist.min()))
 ds=np.array(ds);reports[name]={'sampleCount':len(ds),'hasModelExtrusionWithin025MM':int((ds<=.25).sum()),'misses':int((ds>.25).sum()),'maxDistanceMM':float(ds.max()),'distancesMM':ds.tolist()}
(r/'wall-transect-verification.json').write_text(json.dumps({'method':'At each sampled perimeter position and axial station, check for a non-support extrusion centerline crossing the inner-to-outer wall transect or within0.25mm (approximately half line width), using3Dsegment distance and candidate layers spanning the transect Zextent plus0.13mm. This checks wall presence rather than its possibly empty midpoint. Discrete samples, not a physical print test.','results':reports},indent=2));print({k:{a:b for a,b in v.items()if a!='distancesMM'}for k,v in reports.items()})
