import numpy as np,json,zipfile,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFilter
out=Path('outputs/danya-simulator/assets/projection-prototype/ad5m-one-plate');out.mkdir(exist_ok=True)
r=out.parent/'print';meshes=[];masks=[]
dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
for name,axis,yaw in [('head-upper','Y',0),('head-lower','X',-10)]:
 data=np.fromfile(r/(name+'.stl'),dtype=dtype,offset=84);v=data['v'].astype(float);M=np.array([[0,0,1],[0,1,0],[-1,0,0]])if axis=='Y'else np.array([[1,0,0],[0,0,-1],[0,1,0]]);a=np.deg2rad(yaw);R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]);M=R@M;q=v@M.T;lo=q.min((0,1));q-=lo;dims=np.ceil(q.max((0,1))[:2]*2).astype(int)+11;im=Image.new('L',tuple(dims));draw=ImageDraw.Draw(im)
 for tri in q[:,:,:2]*2+5:draw.polygon([tuple(p)for p in tri],fill=255)
 im=im.filter(ImageFilter.MaxFilter(9));masks.append(np.array(im)>0);meshes.append((name,data,M,lo,q))
a,b=masks;h1,w1=a.shape;h2,w2=b.shape;B=436;conv=np.fft.irfft2(np.fft.rfft2(a,s=(1024,1024))*np.fft.rfft2(b[::-1,::-1],s=(1024,1024)),s=(1024,1024));best=None
for dy in range(12,48):
 for dx in range(196,224):
  w=max(w1,dx+w2);h=max(h1,dy+h2)
  if w<=B and h<=B and conv[dy+h2-1,dx+w2-1]<.1:
   score=max(w,h)+.001*w*h
   if best is None or score<best[0]:best=(score,dx,dy,w,h)
assert best, 'No precise fit';_,dx,dy,w,h=best;offset=np.array([(220-w/2)/2+2.5-110,(220-h/2)/2+2.5-110,0]);info=[];all_data=[]
for i,(name,data,M,lo,q) in enumerate(meshes):
 t=offset+np.array([dx/2,dy/2,0])*i;q+=t;data['v']=q;data['n']=data['n']@M.T;p=out/(name+'-AD5M.stl');p.write_bytes(b'DANYA v12 AD5M rigid print orientation'.ljust(80,b' ')+np.uint32(len(data)).tobytes()+data.tobytes());all_data.append(data);info.append({'part':name,'rotation':M.tolist(),'translation':(t-lo).tolist(),'bounds':[q.min((0,1)).tolist(),q.max((0,1)).tolist()],'sourceSHA256':hashlib.sha256((r/(name+'.stl')).read_bytes()).hexdigest()})
data=np.concatenate(all_data);(out/'DANYA-rear-head-AD5M-plate.stl').write_bytes(b'DANYA upper and lower rear head; keep arrangement'.ljust(80,b' ')+np.uint32(len(data)).tobytes()+data.tobytes());(out/'layout.json').write_text(json.dumps({'bedMM':[220,220,220],'rasterMM':.5,'projectedBufferEachMM':2,'parts':info},indent=2));print(json.dumps(info,indent=2))
im=np.full((440,440,3),245,np.uint8)
for i,mask in enumerate(masks):
 x=int(round((offset[0]+110-2.5)*2))+dx*i;y=int(round((offset[1]+110-2.5)*2))+dy*i;reg=im[y:y+mask.shape[0],x:x+mask.shape[1]];reg[mask]=[44,129,155]if i==0 else[205,132,55]
Image.fromarray(im[::-1]).resize((880,880)).save(out/'plate-footprints.png')
# standards-compliant 3MF with two objects, preserving relative positions
xml=['<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>']
for i,data in enumerate(all_data,1):
 verts,idx=np.unique(data['v'].reshape(-1,3),axis=0,return_inverse=True);xml.append(f'<object id="{i}" type="model" name="{info[i-1]["part"]}"><mesh><vertices>');xml.extend('<vertex x="%.6f" y="%.6f" z="%.6f"/>'%tuple(v)for v in verts);xml.append('</vertices><triangles>');xml.extend('<triangle v1="%d" v2="%d" v3="%d"/>'%tuple(t)for t in idx.reshape(-1,3));xml.append('</triangles></mesh></object>')
xml.append('</resources><build><item objectid="1"/><item objectid="2"/></build></model>')
with zipfile.ZipFile(out/'DANYA-rear-head-AD5M-plate.3mf','w',zipfile.ZIP_DEFLATED)as z:
 z.writestr('3D/3dmodel.model',''.join(xml));z.writestr('[Content_Types].xml','<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>');z.writestr('_rels/.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
