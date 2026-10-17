import numpy as np,json,time
from PIL import Image,ImageDraw,ImageFilter
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype/head-split-quarter/print');out=Path('work/head-split-quarter');B=220
masks=[]
for n in ['head-upper','head-lower']:
 v=np.fromfile(r/(n+'.stl'),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'];lo=v[:,:,:2].min((0,1));q=v[:,:,:2]-lo+5;size=np.ceil(q.max((0,1))).astype(int)+6;im=Image.new('L',tuple(size));d=ImageDraw.Draw(im)
 for tri in q:d.polygon([tuple(p)for p in tri],fill=255)
 im=im.crop(im.getbbox());im.save(out/(n+'-footprint.png'));masks.append(im)
print('footprints',[m.size for m in masks],flush=True)
rot=[]
for im in masks:
 arr=[]
 for deg in range(0,360,5):
  p=im.rotate(deg,expand=True,resample=Image.Resampling.NEAREST);p=p.crop(p.getbbox());p=Image.fromarray(np.pad(np.array(p),4)).filter(ImageFilter.MaxFilter(7));p=p.crop(p.getbbox());arr.append((deg,np.asarray(p)>0))
 rot.append(arr)
results=[];start=time.time()
for aa,a in rot[0][:18]:
 h1,w1=a.shape
 if max(a.shape)>B:continue
 fa=np.fft.rfft2(a,s=(512,512))
 for bb,b in rot[1]:
  h2,w2=b.shape
  if max(b.shape)>B:continue
  # Full convolution: sample index x = dx + w2 - 1.
  conv=np.fft.irfft2(fa*np.fft.rfft2(b[::-1,::-1],s=(512,512)),s=(512,512))
  dx=np.arange(w1-B,B-w2+1);dy=np.arange(h1-B,B-h2+1)
  if not len(dx)or not len(dy):continue
  sums=conv[(dy+h2-1)[:,None],(dx+w2-1)[None,:]];ok=np.argwhere(sums<.1)
  for yy,xx in ok:
   x,y=int(dx[xx]),int(dy[yy]);w=max(w1,x+w2)-min(0,x);h=max(h1,y+h2)-min(0,y);results.append({'a':aa,'b':bb,'dx':x,'dy':y,'w':w,'h':h,'score':max(w,h)+.001*w*h})
 if results:print(aa,'solutions',len(results),flush=True)
results.sort(key=lambda r:r['score']);(out/'packing.json').write_text(json.dumps({'bedMM':B,'rasterMM':1,'bufferEachMM':3,'candidates':results[:30]},indent=2));print('best',results[:3],'seconds',time.time()-start,flush=True)
if results:
 p=results[0];a=rot[0][p['a']//5][1];b=rot[1][p['b']//5][1];x0=max(0,-p['dx']);y0=max(0,-p['dy']);im=np.full((B,B,3),245,dtype=np.uint8)
 for mask,x,y,col in [(a,x0,y0,[48,121,140]),(b,x0+p['dx'],y0+p['dy'],[210,126,50])]:region=im[y:y+mask.shape[0],x:x+mask.shape[1]];region[mask]=col
 Image.fromarray(im).resize((880,880)).save(out/'packing.png')
