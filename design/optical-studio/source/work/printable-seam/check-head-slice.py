import json,re,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'printable-seam';profiles=json.loads((out/'print-transforms.json').read_text());p=np.load(r/'face-v12/perimeter-paths.npz');paths=p['paths'];dirs=p['dirs'];S,N=paths.shape[:2];inside=paths-dirs[None,:,:]*np.linspace(.25,.85,S)[:,None,None];tangent=np.roll(inside,-1,axis=1)-np.roll(inside,1,axis=1);axial=np.gradient(inside,axis=0);normal=np.cross(tangent,axial);normal/=np.maximum(np.linalg.norm(normal,axis=2)[:,:,None],1e-9);factor=np.abs(np.sum(normal*dirs[None,:,:],axis=2));radial=1.4/np.maximum(factor,.45);outside=inside+dirs[None,:,:]*radial[:,:,None];center=np.array([0,0,160]);rho=radial.copy();expanded=np.max(np.stack([np.roll(rho,k,axis=1)for k in range(-16,17)]),axis=0);smooth=sum(np.roll(expanded,k,axis=1)for k in range(-8,9))/17;outside+=dirs[None,:,:]*(smooth-rho)[:,:,None]
results={};full=[]
for name in ['head-upper','head-lower']:
 gcode=Path('work/printable-seam/slices')/name/'plate_1.gcode';layers={};pos=np.zeros(3);feature='Custom';mins=np.ones(3)*1e9;maxs=-mins.copy()
 for line in gcode.open():
  if line.startswith('; FEATURE:'):feature=line.split(':',1)[1].strip()
  if not line.startswith(('G0 ','G1 ')):continue
  vals={k:float(v)for k,v in re.findall(r'([XYZE])(-?[\d.]+)',line.split(';')[0])};q=np.array([vals.get(k,pos[i])for i,k in enumerate('XYZ')])
  if vals.get('E',0)>0 and feature!='Custom' and np.linalg.norm(q[:2]-pos[:2])>0:
   mins=np.minimum(mins,np.minimum(q,pos));maxs=np.maximum(maxs,np.maximum(q,pos))
   if not feature.startswith('Support') and feature not in ['Skirt','Brim','Wipe tower']:
    layers.setdefault(round(q[2],3),[]).append([*pos[:2],*q[:2]])
  pos=q
 layers={z:np.array(v)for z,v in layers.items()};zs=np.array(list(layers));h=np.load(out/'head-band-samples.npz');sample=((h['inside']+h['outside'])/2)[[1,3],::8].reshape(-1,3);sample=sample[sample[:,2]>107.1]if name=='head-upper'else sample[sample[:,2]<106.3];sample=sample@np.array(profiles[name]['R']).T+np.array(profiles[name]['t']);dist=[]
 for q in sample:
  near=zs[np.abs(zs-q[2])<=.13];seg=np.concatenate([layers[z]for z in near])if len(near)else np.zeros((0,4))
  if not len(seg):dist.append(999);continue
  a=seg[:,:2];d=seg[:,2:]-a;u=np.clip(np.sum((q[:2]-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-15),0,1);dist.append(float(np.linalg.norm(a+u[:,None]*d-q[:2],axis=1).min()))
 d=np.array(dist);results[name]={'samples':len(d),'within065MM':int((d<=.65).sum()),'missing':int((d>.65).sum()),'maxDistanceMM':float(d.max()),'distancesMM':dist,'extrusionBoundsMM':[mins.tolist(),maxs.tolist()],'insideAD5MWith025HalfWidth':bool(np.all(mins[:2]-.25>=-110)&np.all(maxs[:2]+.25<=110)&(maxs[2]<=220))};full.append((name,layers))
report={'scope':'Mid-wall samples (90 perimeter positions x2 stations split by head height; exclude0.4mm nearcutplane). Nearest non-support extrusion centerline within0.65mm and layer Zwithin0.13mm. Geometric slicer check, not physical print proof. Same AD5M0.4/0.2mm preset.',**results};(out/'head-slice-verification.json').write_text(json.dumps(report,indent=2));print({n:{k:v for k,v in d.items()if k!='distancesMM'}for n,d in results.items()})
im=Image.new('RGB',(1200,580),'#f3f6f8');draw=ImageDraw.Draw(im)
for col,(name,layers)in enumerate(full):
 z=min(layers,key=lambda z:abs(z-25));draw.text((col*600+20,15),f'{name}: layer Z={z:.2f}mm / model extrusion only',fill='black');draw.rectangle((col*600+25,45,col*600+575,595),outline='#cad4d9')
 for a,b,c,d in layers[z]:draw.line([(col*600+300+a*2.4,315-b*2.4),(col*600+300+c*2.4,315-d*2.4)],fill='#087e8b',width=1)
im.save(out/'head-slice-layers.png')
im=Image.new('RGB',(1120,250),'#f3f6f8');draw=ImageDraw.Draw(im)
for row,(name,d)in enumerate(results.items()):
 draw.text((20,10+row*120),f'{name}: {d["within065MM"]}/{d["samples"]} sample points have extrusion',fill='black')
 for j,val in enumerate(d['distancesMM']):
  station,theta=divmod(j,90);x=20+theta*12;y=35+row*120+station*22;draw.rectangle((x,y,x+10,y+18),fill='#157f89'if val<=.65 else'#de6551')
im.save(out/'head-slice-coverage.png')
