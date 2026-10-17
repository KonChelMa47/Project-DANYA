exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/head-connectors-v3');parts=[];pads=json.load(open('work/buildable-bust-v8/head-pads.json'));interfaces=[]
def rod(p,q,r):
 d=[q[i]-p[i]for i in range(3)];return cy(r,p,d,math.sqrt(sum(v*v for v in d)))
for i,pad in enumerate(pads):
 p=pad['point'].copy();p[2]+=8;n=pad['normal'];u=[-n[1],n[0],0];m=[*u,0,0,0,1,0,*n,0,*p,1]
 interfaces.append({'id':i,'name':pad['name'],'point':p,'normal':n,'matrix':m,'mountPlaneLocalZ':-5,'bossWidthMM':18,'bossHeightMM':12,'screw':'M3x8','washerCount':3,'washerThicknessMM':.5,'nutPocketAFMM':5.8,'nutPocketDepthMM':2.65})
 for j in range(3):sv(f'head-connect-{i}-washer-{j}',cut(cyl(3.5,-9.5+.5*j,.5),cyl(1.7,-9.6+.5*j,.7)),'roll',m,'purchased M3 washer OD7 t0.5','Three0.5mmwashers perjoint tolimit screwpenetration.')
 sv(f'head-connect-{i}-bolt',fuse(cyl(1.5,-9.5,8),cyl(2.75,-12.5,3)),'roll',m,'purchased M3x8 socket screw','Shaft stops atlocal-1.5,blindbore-1.3; actualstacktolerances requirecheck.')
 sv(f'head-connect-{i}-nut',cut(hexagon(5.5,-4.9,2.4),cyl(1.3,-5,2.6)),'roll',m,'purchased M3 hex nut AF5.5 t2.4','Rearloadedcaptivenut,retained bysupportflange; no tappedplastic.')
for sg in [-1,1]:
 tag='left'if sg<0 else'right';post=[-35,sg*40,78]
 for mode in ['static','motorized']:
  if mode=='static':
   a=box(-114,sg*85-12,26,48,24,8);a=fuse(a,rod([-90,sg*85,30],[-35,sg*40,30],7));a=fuse(a,rod([-35,sg*40,28],post,5));a=cut(a,box(-500,-500,-500,1000,1000,526))
   for x in [-106,-74]:a=cut(a,cyl(2.2,25.9,8.2,x,sg*85))
  else:
   hy=-38 if sg<0 else 21;py=-32 if sg<0 else 30
   reader=STEPControl_Reader();reader.ReadFile(str((out/'archive-motorized'/('trial-'+tag+'-head-rail.step')) if (out/'archive-motorized'/('trial-'+tag+'-head-rail.step')).exists()else Path('outputs/danya-simulator/assets/buildable-bust-v10')/('trial-'+tag+'-head-rail.step')));reader.TransferRoots();a=move(reader.OneShape(),[1,0,0,0,0,1,0,0,0,0,1,0,0,0,8,1]);a=cut(a,box(-500,-500,55.5,1000,1000,500));stem=rod([-48,py,55],[-35,sg*40,61],4.5);stem=cut(stem,box(-500,-500,-500,1000,1000,555));a=fuse(a,stem);a=fuse(a,rod([-35,sg*40,57],post,5));a=cut(a,cyl(1.7,50.9,3.2,-48,hy));a=cut(a,cyl(3.7,54,5,-48,hy))
  for d in interfaces:
   if d['point'][1]*sg<0:continue
   m=d['matrix'];endpoint=point([-6.5*sg,0,-6.5],m);q=[-35,sg*40,d['point'][2]]
   branch=fuse(rod(q,endpoint,3.5),move(box(-9,-6,-8,18,12,3),m));branch=cut(branch,move(box(-40,-20,-5,80,40,40),m));a=fuse(a,branch)
   a=cut(a,move(cyl(1.7,-25,24,0,0),m));a=cut(a,move(cyl(3.7,-25,17,0,0),m))
  save(f'{mode}-head-arm-{tag}',a,'roll',[0,0],'printed PETG candidate','FourcommonM3x8headconnections. Twoorone rootfastenerspermode. Integralheadbossesholdnuts. No glue requiredforheadattachment. Physicalstrengthuntested.')
(out/'connectors.json').write_text(json.dumps({'parts':parts,'interfaces':interfaces,'coordinates':'neutral world withcommon8mmlift','status':'candidate untilmesh/optical/interference checks'},indent=2));print('GENERATED',len(parts))
