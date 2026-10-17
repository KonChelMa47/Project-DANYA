exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
right=next(r for r in json.load(open('work/buildable-bust-v10/final-optical-route-probes.json'))['rows']if r['name']=='projector-right-revised');left=next(r for r in json.load(open('work/buildable-bust-v10/left-wide-detours.json'))['rows']if all(not c['hits']and not c['lightHits']for c in r['checks']));branch=next(r for r in json.load(open('work/buildable-bust-v10/fold-alternate-probes.json'))['rows']if len(r['checks'])==3 and all(not c['hits']and not c['lightHits']for c in r['checks']))
assert all(not c['hits']and not c['lightHits']for c in right['checks'])
def chain(pts):
 pts=[[x,y,z-8]for x,y,z in pts];a=None
 for u,v in zip(pts,pts[1:]):
  delta=[v[i]-u[i]for i in range(3)];b=cy(3,u,delta,math.sqrt(sum(x*x for x in delta)));a=b if a is None else fuse(a,b)
 for p in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*p),3).Shape())
 return a
def rootfoot(x):return fuse(fuse(cyl(4,34,3,x,-38),cyl(4,34,3,x,-29)),box(x-4,-38,34,8,9,3))
# All CAD stored before common8mm upper lift. The optical drop is baked here.
pm=dat['projector']['matrix'].copy();pm[12:15]=point([-111.62,-13.97,-81.33583308323333],pm);pm[14]-=6
fm=dat['foldMirror']['matrix'].copy();fm[14]-=6
for role,row,x,zz in [('left',left,-48,68)]:
 a=fuse(chain(row['points']),rootfoot(x));a=cut(a,cyl(1.7,33.9,3.2,x,-38));a=fuse(a,move(cy(5,[123,-11,zz],[0,1,0],3),pm));a=cut(a,move(cy(1.7,[123,-30,zz],[0,1,0],22.1),pm));a=cut(a,move(cy(3.7,[123,-30,zz],[0,1,0],19),pm))
 if role=='left':
  a=fuse(a,chain(branch['points']));a=fuse(a,move(box(-9,-2.3,-21.6,18,4.6,6),fm))
  for xx in [-5,5]:a=cut(a,move(cyl(1.7,-35,20,xx,0),fm))
  o=layout['foldMirror'];w=o['width'];h=o['height']
  for xx,yy in [(-.25*w,-.18*h),(.25*w,-.18*h),(0,.25*h)]:a=cut(a,move(cyl(3.7,-19.3,3.8,xx,yy),fm));a=cut(a,move(cyl(1.8,-40,20.8,xx,yy),fm))
 rr=STEPControl_Reader();assert rr.ReadFile(str(out/'drop-projector-tray-t5.step'))==1;rr.TransferRoots();a=cut(a,rr.OneShape())
 save('revised-projector-support-'+role+'-central',a,'roll',[0,0],'metal6mm support envelope','Collision-routed optical support; complete print/stock-joint conversion and strength qualification pending. Do not print thin rods as load-bearing polymer. M3 receiver lug and carrier through-bolt interface included.')
(out/'central-optical-support-candidate.json').write_text(json.dumps({'parts':parts,'branch':branch},indent=2));print('VALID',len(parts))
