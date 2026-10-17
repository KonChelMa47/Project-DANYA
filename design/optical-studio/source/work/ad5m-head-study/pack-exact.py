exec(open('work/ad5m-head-study/pack-tilts.py').read().split('models=[]')[0])
models=[]
for name in ['head-upper','head-lower']:
 v=np.fromfile(r/(name+'.stl'),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'];variants=[]
 for axis,deg in [('Y',90),('X',90),('Y',75),('Y',-75),('X',75),('X',-75),('Y',60),('Y',-60),('X',60),('X',-60)]:
  a=np.deg2rad(deg);c,s=np.cos(a),np.sin(a);M=np.array([[1,0,0],[0,c,-s],[0,s,c]])if axis=='X'else np.array([[c,0,s],[0,1,0],[-s,0,c]]);q=v@M.T;lo=q[:,:,:2].min((0,1));dims=np.ceil(np.ptp(q[:,:,:2].reshape(-1,2),axis=0)).astype(int)+1;im=Image.new('L',tuple(dims));d=ImageDraw.Draw(im)
  for tri in q[:,:,:2]-lo:d.polygon([tuple(p)for p in tri],fill=255)
  opts=[]
  for yaw in range(0,360,5):
   p=im.rotate(yaw,expand=True);p=p.crop(p.getbbox());p=Image.fromarray(np.pad(np.array(p),3)).filter(ImageFilter.MaxFilter(5));opts.append(np.asarray(p)>0)
  variants.append({'axis':axis,'tilt':deg,'matrix':M.tolist(),'opts':opts});print(name,axis,deg,flush=True)
 models.append(variants)
B=218
for ia,A in enumerate(models[0]):
 for ib,C in enumerate(models[1]):
  for aa in range(18):
   a=A['opts'][aa];h1,w1=a.shape;fa=np.fft.rfft2(a,s=(512,512))
   for bb,b in enumerate(C['opts']):
    h2,w2=b.shape;dx=np.arange(w1-B,B-w2+1);dy=np.arange(h1-B,B-h2+1)
    if len(dx)==0 or len(dy)==0:continue
    conv=np.fft.irfft2(fa*np.fft.rfft2(b[::-1,::-1],s=(512,512)),s=(512,512));ok=np.argwhere(conv[(dy+h2-1)[:,None],(dx+w2-1)[None,:]]<.1)
    if len(ok):
     j,i=ok[len(ok)//2];x,y=int(dx[i]),int(dy[j]);data={'ia':ia,'ib':ib,'yaw':[aa*5,bb*5],'dx':x,'dy':y,'tilts':[A['tilt'],C['tilt']],'axes':[A['axis'],C['axis']]};print('FIT',data,flush=True);(out/'exact-packing.json').write_text(json.dumps(data,indent=2));np.savez(out/'exact-masks.npz',a=a,b=b);raise SystemExit
  print('pair',ia,ib,flush=True)
