exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
import shutil
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[];probe=json.load(open('work/buildable-bust-v10/deep-rear-lower-probes.json'));assert all(not c['hits']and not c['lightHits']for r in probe['rows']for c in r['checks'])
def tube(a,b,r=5.6):
 d=[b[i]-a[i]for i in range(3)];return cy(r,a,d,math.sqrt(sum(v*v for v in d)))
def chain(pts,r=5.6):
 a=tube(pts[0],pts[1],r)
 for p in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*p),r).Shape())
 for i in range(2,len(pts)):a=fuse(a,tube(pts[i-1],pts[i],r))
 return a
routes={r['name']:r['points']for r in probe['rows']};a=chain(routes['cross-lower'])
for name in ['spine0','spine1','base-left','base-rear','motor','foot0','foot1']:a=fuse(a,chain(routes[name]))
for yy in [-60,60]:a=fuse(a,cyl(9,-152.2,6,-55,yy));a=cut(a,cyl(2.2,-152.3,6.2,-55,yy))
for name in ['base-left','base-rear']:
 x,y,z=routes[name][0];a=fuse(a,cyl(4.5,-33,5,x,y));a=cut(a,cyl(1.7,-55,28,x,y))
drive=json.load(open(out/'trial-drive-layout.json'));d=drive['motors'][0];d['quaternion']=probe['quaternion'];m=qm(d['quaternion'],d['tip']);a=fuse(a,move(box(12,-48.6,-10,17,6,20),m))
for z in [-5,5]:a=cut(a,move(cy(1.7,[20.5,-75,z],[0,1,0],32.5),m));a=cut(a,move(cy(3.2,[20.5,-75,z],[0,1,0],28.4),m))
a=cut(a,box(-300,-300,-28,600,600,200));a=cut(a,box(-300,-300,-300,600,600,147.8))
save('trial-rear-chassis',a,'fixed',[0,0],'printed PETG candidate','Rear cage inside3mm shell, crossbeam atZ-118 avoids swept pitch belt. TwoM3 bearing-base and twoM4 floor attachments. Integral yaw rear mount, unchanged gear datum. Material/print-orientation/2kg proof qualification pending.')
for row in json.load(open(out/'motor-backplates-v10.json'))['parts']:
 if 'backmount-yaw'not in row['name']:continue
 new=dict(row,name=row['name'].replace('selected-','trial-'));parts.append(new)
 for ext in ['stl','step']:shutil.copy2(out/(row['name']+'.'+ext),out/(new['name']+'.'+ext))
for i,z in enumerate([-5,5]):sv('trial-yaw-bracket-bolt-'+str(i),fuse(cy(1.5,[20.5,-46.6,z],[0,1,0],8),cy(2.75,[20.5,-49.6,z],[0,1,0],3)),'fixed',m,'M3x8','Envelope, rear captive nut stack.')
pts=[(-8+77*math.cos(i*math.pi/48),.345745+105*math.sin(i*math.pi/48))for i in range(96)];a=poly(pts,-159,6.8);a=cut(a,box(-37,-35,-159.1,60,70,7))
for x,y in [(-55,-60),(-55,60),(-75,0),(60,0),(-10,-98),(-10,98)]:a=cut(a,cyl(2.2,-159.1,7,x,y))
save('trial-floor',a,'fixed',[0,0],'printed PETG candidate','6.8mm floor within existingbody height, bottomZ-159 topZ-152.2. 154x210mm outline fits inside3mm wall candidate. Used only with8mm upper-assembly lift. Shell mounting bosses and physical proof test pending.')
(out/'deep-rear-cage.json').write_text(json.dumps({'status':'candidate for lifted upper mechanism','liftMM':8,'parts':parts},indent=2));(out/'trial-drive-layout.json').write_text(json.dumps(drive,indent=2));p=out/'trial-structural-candidate.json';data=json.load(open(p));names=[r['name']for r in parts];data['parts']=[r for r in data['parts']if r['name']not in names]+parts;p.write_text(json.dumps(data,indent=2));print('VALID',len(parts))
