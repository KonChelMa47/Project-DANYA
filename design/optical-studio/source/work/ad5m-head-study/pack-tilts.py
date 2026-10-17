import numpy as np,json,time
from PIL import Image,ImageDraw,ImageFilter
from pathlib import Path
out=Path('work/ad5m-head-study');r=Path('outputs/danya-simulator/assets/projection-prototype/print');B=218

def hull(q):
 ps=sorted(set(map(tuple,np.round(q,2))));lo=[];hi=[]
 def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
 for p in ps:
  while len(lo)>1 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
  lo.append(p)
 for p in ps[::-1]:
  while len(hi)>1 and cross(hi[-2],hi[-1],p)<=0:hi.pop()
  hi.append(p)
 return np.array(lo[:-1]+hi[:-1])
models=[]
for name in ['head-upper','head-lower']:
 v=np.fromfile(r/(name+'.stl'),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'].reshape(-1,3);v=np.unique(v,axis=0);vs=v[::4];variants=[]
 for axis in ['X','Y']:
  for deg in [0,30,45,60,75,90]:
   if axis=='X'and deg==0:continue
   for sign in ([1]if deg in [0,90]else[-1,1]):
    a=np.deg2rad(deg*sign);c,s=np.cos(a),np.sin(a);M=np.array([[1,0,0],[0,c,-s],[0,s,c]])if axis=='X'else np.array([[c,0,s],[0,1,0],[-s,0,c]]);q=vs@M.T;poly=hull(q[:,:2]);variants.append({'axis':axis,'tilt':deg*sign,'matrix':M.tolist(),'poly':poly,'height':float(np.ptp(q[:,2]))})
 models.append(variants)
print('variants',len(models[0]),flush=True)
# Try modest orientations first. Search convex outlines with 3mm clearance buffer per part.
rows=[];start=time.time()
for ia,A in enumerate(models[0]):
 for ib,C in enumerate(models[1]):
  if max(abs(A['tilt']),abs(C['tilt']))<30:continue
  options=[]
  for src in [A,C]:
   rr=[]
   for yaw in range(0,360,15):
    a=np.deg2rad(yaw);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);q=src['poly']@R.T;lo=q.min(0);dims=np.ceil(np.ptp(q,axis=0)).astype(int)+9
    if max(dims)>B:rr.append(None);continue
    im=Image.new('L',tuple(dims));ImageDraw.Draw(im).polygon([tuple(p)for p in q-lo+4],fill=1);im=im.filter(ImageFilter.MaxFilter(7));arr=np.asarray(im)>0;rr.append((yaw,arr))
   options.append(rr)
  best=None
  for aa in range(0,6):
   if options[0][aa]is None:continue
   yaw,a=options[0][aa];h1,w1=a.shape;fa=np.fft.rfft2(a,s=(512,512))
   for bb in range(24):
    if options[1][bb]is None:continue
    yy,b=options[1][bb];h2,w2=b.shape;dx=np.arange(w1-B,B-w2+1);dy=np.arange(h1-B,B-h2+1)
    if len(dx)==0 or len(dy)==0:continue
    conv=np.fft.irfft2(fa*np.fft.rfft2(b[::-1,::-1],s=(512,512)),s=(512,512));sub=conv[(dy+h2-1)[:,None],(dx+w2-1)[None,:]];ok=np.argwhere(sub<.1)
    if len(ok):
     for j,i in ok:
      x,y=int(dx[i]),int(dy[j]);w=max(w1,x+w2)-min(0,x);h=max(h1,y+h2)-min(0,y);score=max(w,h)
      if best is None or score<best['size']:best={'ia':ia,'ib':ib,'aYaw':yaw,'bYaw':yy,'dx':x,'dy':y,'size':score,'width':w,'height':h,'tilts':[A['tilt'],C['tilt']],'axes':[A['axis'],C['axis']]}
  if best:
   rows.append(best);print('FIT',best,flush=True)
   # Found a stable-sized candidate; retain variants so exact coordinates can be reconstructed.
   if best['size']<=216:
    payload={'candidate':best,'variants':[[{k:v for k,v in z.items()if k!='poly'}for z in m]for m in models]};(out/'tilt-packing.json').write_text(json.dumps(payload,indent=2));print('elapsed',time.time()-start,flush=True);raise SystemExit
 print('upper variant',ia,'seconds',time.time()-start,flush=True)
(out/'tilt-packing.json').write_text(json.dumps({'candidates':rows},indent=2))
