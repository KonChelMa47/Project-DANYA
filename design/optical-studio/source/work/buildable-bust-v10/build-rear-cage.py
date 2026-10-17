exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
def tube(a,b,r=5.6):
 d=[b[i]-a[i]for i in range(3)];return cy(r,a,d,math.sqrt(sum(v*v for v in d)))
def chain(pts,r=5.6):
 a=tube(pts[0],pts[1],r)
 for p in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*p),r).Shape())
 for i in range(2,len(pts)):a=fuse(a,tube(pts[i-1],pts[i],r))
 return a
spines=[r['points'][:-1] for r in json.load(open('work/buildable-bust-v10/back-chassis-probes.json')) if r['yy']==60]
cross=json.load(open('work/buildable-bust-v10/back-negative-probes.json'))[:8];crosspts=[cross[0]['start']]+[r['end']for r in cross]
spines=[[[p[0]+1.5,p[1],p[2]]for p in points]for points in spines]
crosspts=[[p[0]+1.5,p[1],p[2]]for p in crosspts]
a=chain(crosspts)
for points in spines:
 a=fuse(a,chain(points));foot=[-58,points[0][1],-156];a=fuse(a,chain([points[1],points[0],foot]));a=fuse(a,cyl(9,-159,6,*foot[:2]));a=cut(a,cyl(2.2,-159.1,6.2,*foot[:2]))
# Two separated bearing-base mounting lugs. Connection remains a load-test candidate.
for start,end in [([-49.016168594,-46.4196267,-31],spines[0][-1]),([-76.016168594,.345745,-31],crosspts[4])]:
 # rear attachment initially stays under base; validate against motor housing before adoption.
 a=fuse(a,chain([start,end]));a=fuse(a,cyl(6,-33,5,*start[:2]));a=cut(a,cyl(1.7,-33.1,5.2,*start[:2]))
cage=a
save('trial-rear-chassis',a,'fixed',[0,0],'printed PETG candidate','Two rear conformal spines and curved crossbeam. Two M4 floor attachments and two M3 bearing-base attachments. FDM section orientation, deflection and 2kg proof load unqualified. No metal fabrication assumed for this part.')
# Reorient only yaw housing around its unchanged output datum. Keep all driven gears fixed.
drive=json.load(open(out/'drive-layout.json'));old=drive['motors'][0];q=json.load(open('work/buildable-bust-v10/yaw120-probes.json'))['quaternion'];new=dict(old,quaternion=q);drive['motors'][0]=new;(out/'trial-drive-layout.json').write_text(json.dumps(drive,indent=2));m=qm(q,new['tip'])
def tr(mat):
 t=gp_Trsf();t.SetValues(mat[0],mat[4],mat[8],mat[12],mat[1],mat[5],mat[9],mat[13],mat[2],mat[6],mat[10],mat[14]);return t
oldtr=tr(qm(old['quaternion'],old['tip']));newtr=tr(m);delta=newtr.Multiplied(oldtr.Inverted())
for row in json.load(open(out/'motor-backplates-v10.json'))['parts']:
 if 'backmount-yaw' not in row['name']:continue
 rr=STEPControl_Reader();rr.ReadFile(str(out/(row['name']+'.step')));rr.TransferRoots();a=BRepBuilderAPI_Transform(rr.OneShape(),delta,True).Shape();save(row['name'].replace('selected-','trial-'),a,'fixed',[0,0],row['material'],row['notes'])
a=box(12,-48.6,-10,17,6,20)
for z in [-5,5]:a=cut(a,cy(1.7,[20.5,-48.7,z],[0,1,0],6.2));a=cut(a,cy(3.1,[20.5,-48.7,z],[0,1,0],2.1))
a=move(a,m);start=point([20.5,-50,0],m);a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*start),5.6).Shape());a=fuse(a,chain([start,[-103.5,0,-73],crosspts[4]]))
# Continuous screwdriver/head clearance through the supporting root.
for z in [-5,5]:a=cut(a,move(cy(3.2,[20.5,-75,z],[0,1,0],28.4),m))
a=fuse(cage,a);parts=[p for p in parts if p['name']!='trial-rear-chassis']
save('trial-rear-chassis',a,'fixed',[0,0],'printed PETG candidate','One-piece rear chassis and servo bracket. Integral connection, two M3x8 servo frame bolts with continuous head/tool clearance. Two M3 bearing-base and two M4 floor attachments. 2kg load qualification pending.')
for i,z in enumerate([-5,5]):
 a=fuse(cy(1.5,[20.5,-46.6,z],[0,1,0],8),cy(2.75,[20.5,-49.6,z],[0,1,0],3));sv('trial-yaw-bracket-bolt-'+str(i),a,'fixed',m,'M3x8','Envelope')
# Lower printed floor: motion gap without moving optical datums.
pts=[(-7.67+80*math.cos(i*math.pi/48),.3457+109*math.sin(i*math.pi/48))for i in range(96)]
a=poly(pts,-167,6.8);a=cut(a,box(-37,-35,-167.1,60,70,8.2))
for yy in [-60,60]:a=fuse(a,cyl(9,-160.2,1.2,-58,yy))
for x,y in [(-58,-60),(-58,60),(-75,0),(60,0),(-10,-98),(-10,98)]:a=cut(a,cyl(2.2,-167.1,8.2,x,y))
save('trial-floor',a,'fixed',[0,0],'printed PETG candidate','6.8mm floor plate below moving envelope: topZ-160.2 bottomZ-167, two1.2mm locating risers. Adds7mm below former body bottom. Existing shell/base fastening interface needs final qualification.')
(out/'trial-rear-cage.json').write_text(json.dumps({'status':'route candidate, not released','parts':parts},indent=2))
base=json.load(open(out/'structural-candidate.json'));base['parts']=[p for p in base['parts']if not p['name'].startswith(('selected-backmount-yaw','yaw-motor-rear-stand','yaw-rear-stand-bolt','yaw-floor-')) and p['name']not in ['frame-base-left','frame-base-right','frame-base-third','frame-floor']]+parts;base['status']='REAR CAGE TRIAL: connection details and full collision validation pending';(out/'trial-structural-candidate.json').write_text(json.dumps(base,indent=2))
print('VALID',len(parts))
