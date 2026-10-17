import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
root=Path(__file__).parent
im=Image.new('RGB',(1400,1000),'white'); dr=ImageDraw.Draw(im)
specs=[(1,0,0,2,'Midline y=0',0,0),(2,100,0,1,'Horizontal z=100',700,0),(2,170,0,1,'Horizontal z=170',0,500),(2,210,0,1,'Horizontal z=210',700,500)]
for name,col in [('back','#3366cc'),('shell','#ee5533')]:
 d=np.load(root/f'DANYA-face-{name}.npz'); v=d['v']; f=d['loops'].reshape(-1,3); tri=v[f]
 for axis,level,ax,ay,label,ox,oy in specs:
  t=tri[(tri[:,:,axis].min(1)<=level)&(tri[:,:,axis].max(1)>=level)]
  for q in t:
   pts=[]
   for i,j in [(0,1),(1,2),(2,0)]:
    a,b=q[i],q[j]
    if (a[axis]-level)*(b[axis]-level)<0:
     p=a+(b-a)*(level-a[axis])/(b[axis]-a[axis]); pts.append((ox+330+2.0*p[ax],oy+480-1.9*p[ay] if ay==2 else oy+250-2*p[ay]))
   if len(pts)==2: dr.line(pts,fill=col,width=2)
  dr.text((ox+10,oy+10),label+' | blue=back red=face',fill='black')
im.save(root/'sections.png')
