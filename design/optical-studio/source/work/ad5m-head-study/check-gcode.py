import re,json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('outputs/danya-simulator/assets/projection-prototype/ad5m-one-plate');p=Path('work/ad5m-head-study/sliced/plate_1.gcode');bounds={};segments=[];pos={'X':0.,'Y':0.,'Z':0.};feat='Custom';relative=True
for line in p.open():
 if line.startswith('; FEATURE:'):feat=line.split(':',1)[1].strip()
 if line.startswith('M83'):relative=True
 if line.startswith('M82'):relative=False
 if line.startswith(('G0 ','G1 ')):
  vals={k:float(v)for k,v in re.findall(r'([XYZE])(-?[\d.]+)',line.split(';')[0])};new={k:vals.get(k,v)for k,v in pos.items()}
  if vals.get('E',0)>0 and feat!='Custom' and any(new[k]!=pos[k]for k in ['X','Y']):
   a=np.array(list(pos.values()));b=np.array(list(new.values()));mn=np.minimum(a,b);mx=np.maximum(a,b)
   if feat not in bounds:bounds[feat]=[mn,mx,0]
   else:bounds[feat][0]=np.minimum(bounds[feat][0],mn);bounds[feat][1]=np.maximum(bounds[feat][1],mx)
   bounds[feat][2]+=1;segments.append((a,b,feat))
  pos=new
mins=np.min([b[0]for b in bounds.values()],axis=0);maxs=np.max([b[1]for b in bounds.values()],axis=0);report={'extrusionCenterlineBoundsMM':[mins.tolist(),maxs.tolist()],'xyEnvelopeWithHalfLineWidthMM':[(mins[:2]-.25).tolist(),(maxs[:2]+.25).tolist()],'bedCenteredMM':[-110,110],'insideWithHalfLineWidth':bool(np.all(mins[:2]-.25>=-110)and np.all(maxs[:2]+.25<=110)and maxs[2]<=220),'features':{k:{'min':v[0].tolist(),'max':v[1].tolist(),'segments':v[2]}for k,v in bounds.items()},'physicalPrintTest':False}
(root/'slice-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
for name,zlim in [('support-footprint',999),('first-layer',.21)]:
 im=Image.new('RGB',(1000,1000),'#f4f6f8');d=ImageDraw.Draw(im);d.rectangle((38,38,962,962),fill='#e1e5e9',outline='#444444',width=2)
 for a,b,f in segments:
  if max(a[2],b[2])>zlim:continue
  xy=[(float(v[0]*4.2+500),float(500-v[1]*4.2))for v in [a,b]];d.line(xy,fill='#d98841'if f.startswith('Support')else'#277e96',width=1)
 im.save(root/(name+'.png'))
