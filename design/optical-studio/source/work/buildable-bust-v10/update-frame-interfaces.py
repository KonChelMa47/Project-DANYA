exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
pm=dat['projector']['matrix'].copy();pm[12:15]=point([-111.62,-13.97,-81.33583308323333],pm)
for old,zz in [('frame-projector-right',7),('frame-optics-left',68)]:
 r=STEPControl_Reader();r.ReadFile(str(out/(old+'.step')));r.TransferRoots();a=r.OneShape()
 if old=='frame-projector-right':
  pts=[[4,-29,44],[-11,-59,-8],[-11,-64,-13],[-16,-79,-91],point([119,-9.5,7],pm)];a=cyl(5,43,3,4,-29)
  for u,v in zip(pts,pts[1:]):
   delta=[v[i]-u[i]for i in range(3)];a=fuse(a,cy(3,u,delta,math.sqrt(sum(x*x for x in delta))))
  for pnt in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*pnt),3).Shape())
  a=cut(a,cyl(1.65,42.9,3.2,4,-29))
 # Replace original tray lug, matched to the measured-base receiver underside.
 a=cut(a,move(box(112,-8,zz-10,25,30,20),pm));a=fuse(a,move(cy(5,[123,-11,zz],[0,1,0],3),pm));a=cut(a,move(cy(1.7,[123,-30,zz],[0,1,0],22.1),pm));a=cut(a,move(cy(3.7,[123,-30,zz],[0,1,0],19),pm))
 if old=='frame-optics-left':
  fm=dat['foldMirror']['matrix'];a=cut(a,move(box(-10,-5,-15.6,20,10,8.5),fm));a=fuse(a,move(box(-9,-2.3,-21.6,18,4.6,6),fm))
  for x in [-5,5]:a=cut(a,move(cyl(1.7,-35,20,x,0),fm))
  o=layout['foldMirror'];w=o['width'];h=o['height']
  for xx,yy in [(-.25*w,-.18*h),(.25*w,-.18*h),(0,.25*h)]:
   a=cut(a,move(cyl(3.7,-19.3,3.8,xx,yy),fm));a=cut(a,move(cyl(1.8,-40,20.8,xx,yy),fm))
  for j,x in enumerate([-5,5]):
   sv('trial-fold-frame-bolt-'+str(j),fuse(cyl(1.5,-22.1,12,x,0),cyl(2.75,-25.1,3,x,0)),'roll',fm,'M3x12','12mm underhead,0.5washer+6mm support+5mm plate. TipZ-10.1 clear of backing at-7.6.')
   sv('trial-fold-frame-washer-'+str(j),cut(cyl(3.5,-22.1,.5,x,0),cyl(1.6,-22.2,.7,x,0)),'roll',fm,'M3 washer OD7 t0.5','Envelope')
 save('trial-'+old,a,'roll',[0,0],'interface geometry candidate','Revised tray/fold mirror interfaces. Retained6mm metal rod route; complete printable or stock-joint fabrication conversion remains pending. Do not print this whole rod assembly as unreinforced polymer.')
 sv('trial-tray-frame-bolt-'+str(zz),fuse(cy(1.5,[123,-11.5,zz],[0,1,0],6),cy(2.75,[123,-14.5,zz],[0,1,0],3)),'roll',pm,'M3x6','0.5washer+3mm lug ->2.5mm engagement; tipY-5.5 inside blind pocket, does not enter projector base.')
 sv('trial-tray-frame-washer-'+str(zz),cut(cy(3.5,[123,-11.5,zz],[0,1,0],.5),cy(1.6,[123,-11.6,zz],[0,1,0],.7)),'roll',pm,'M3 washer OD7 t0.5','Envelope')
(out/'trial-frame-interfaces.json').write_text(json.dumps({'status':'geometric interfaces only, fabrication conversion pending','parts':parts},indent=2));f=out/'trial-structural-candidate.json';d=json.load(open(f));names=[p['name']for p in parts]+['frame-projector-right','frame-optics-left'];d['parts']=[p for p in d['parts']if p['name']not in names]+parts;f.write_text(json.dumps(d,indent=2));print('VALID',len(parts))
