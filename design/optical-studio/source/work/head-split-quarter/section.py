import numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);im=Image.new('RGB',(1400,700),'white');d=ImageDraw.Draw(im)
for j,z in enumerate([112,158]):
 for n,col,path in [('head','#2483a1',r/'head-split-quarter/head-upper.stl'),('carrier','#d98a36',r/'assembly/magnet-carrier.stl'),('face','#bb5287',r/'assembly/face.stl')]:
  v=np.fromfile(path,dtype=dt,offset=84)['v'];v=v[(v[:,:,2].min(1)<=z)&(v[:,:,2].max(1)>=z)]
  for t in v:
   ps=[]
   for a,b in zip(t,np.roll(t,1,axis=0)):
    if (a[2]<z)!=(b[2]<z):ps.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
   if len(ps)==2:d.line([(j*700+(p[0]+10)*24,650-(p[1]+80)*24)for p in ps],fill=col,width=3)
 d.text((j*700+20,20),str(z)+'mm: blue head, orange carrier, pink face',fill='black')
im.save(r/'head-split-quarter/cross-sections.png')
