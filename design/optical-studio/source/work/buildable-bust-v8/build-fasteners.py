exec(open('work/buildable-bust-v8/build-mounts.py').read().split('# Split motor saddles')[0])
from OCP.STEPControl import STEPControl_Reader
parts=json.load(open(out/'parts.json'))['parts'];
for part in parts:
 if part['name'].startswith('projector-base-pad-'):part['material']='aluminium spacer3';part['notes']='Rigid 3mm datum pad under metal base. Upper compliant pads provide retention preload.'
parts=[p for p in parts if not p['name'].startswith(('fastener-','projector-edge-','projector-clamp-'))]
def flatbolt(p,axis,L):return fuse(cy(1.5,p,axis,L),BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(*p),gp_Dir(*axis)),3.1,1.5,1.6).Shape())
def countersink(p,axis):return fuse(cy(1.65,[p[i]-.1*axis[i]for i in range(3)],axis,3.2),BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(*p),gp_Dir(*axis)),3.3,1.65,1.65).Shape())
def bolt(p,axis,L,r=1.5):return fuse(cy(r,p,axis,L),cy(r*1.8,[p[i]-2.5*axis[i]for i in range(3)],axis,2.5))
for d,parent in zip(drive['motors'],['fixed','yaw','pitch']):
 m=qm(d['quaternion'],d['tip'])
 for z in [-15.65,15.65]:sv('fastener-'+d['role']+'-clamp-'+str(z),bolt([18,1.3,z],[0,-1,0],45),parent,m,'steel M3x45','Clamp cap screw; five mm nominal threaded engagement at back of saddle.')
 # Two rear fastening bolts plus backing washer bridge, route rail is connected here.
 a=box(12,-46.5,-4,12,3,8)
 for x in [14.5,21.5]:a=cut(a,cy(1.65,[x,-46.6,0],[0,1,0],3.2))
 sv('fastener-'+d['role']+'-rear-bridge',a,parent,m,'steel3','Two-hole fastening lug between saddle and support rod.')
 for x in [14.5,21.5]:sv('fastener-'+d['role']+'-frame-'+str(x),flatbolt([x,-46.5,0],[0,1,0],8),parent,m,'steel M3x8','3mm bracket, 5mm engagement; clearance between screw tip and servo case.')
for key in ['mirror','foldMirror']:
 o=layout[key];m=dat[key]['matrix'];w,h=o['width'],o['height']
 for j,(x,y)in enumerate([[-.25*w,-.18*h],[.25*w,-.18*h],[0,.25*h]]):
  sv('fastener-'+key+'-adjust-'+str(j),flatbolt([x,y,-14.1],[0,0,1],12),'roll',m,'steel M3x12','Nominal 3mm adjustment gap. Do not reduce gap below3mm: screw tip clearance to glass is1mm. Lock after optical calibration.')
  sv('fastener-'+key+'-elastic-'+str(j),cut(cyl(3,-8.1,3,x,y),cyl(1.65,-8.2,3.2,x,y)),'roll',m,'silicone sleeve','Compliant 3mm sleeve between backing and fixed plate; calibration preload, not spring-rated.')
# Peripheral clips retain the existing metal base without drilling the projector.
m=dat['projector']['matrix'].copy();m[12:15]=point([-111.62,-13.97,-81.33583308323333],m)
p=next(p for p in parts if p['name']=='projector-open-tray');a=box(2,-6,2,131,3,71)
for xx,ww in [(9,43),(60,66)]:a=cut(a,box(xx,-6.1,9,ww,3.2,57))
for xx in [12,123]:
 for zz in [7,68]:a=cut(a,cy(1.65,[xx,-6.1,zz],[0,1,0],3.2))
a=move(a,m)
for x in [16,48,80,96]:
 for z in ([78] if x==96 else [-3]):
  z0=-6 if z<0 else 72;tab=cut(box(x-4,-6,z0,8,3,9),cy(1.65,[x,-6.1,z],[0,1,0],3.2));a=fuse(a,move(tab,m))
  clamp=cut(box(x-4,4.47,z0,8,3,9),cy(1.65,[x,4.37,z],[0,1,0],3.2));name=str(x)+'-'+str(z)
  sv('projector-edge-clip-'+name,clamp,'roll',m,'6061-T6 aluminium3','Peripheral base hold-down. No drilling through projector base or PCB. 0.5mm pad at metal base upper edge, four M3x20 screws.')
  sv('projector-edge-pad-'+name,box(x-4,3.97,0 if z<0 else 72,8,.5,3),'roll',m,'soft pad1 compressed0.5','Upper clamp contact pad; 1mm supply thickness, 0.5mm assembled gap. Determine clamp force by physical test, rigid lower pads preserve lens datum.')
  spacer=cut(cy(2.8,[x,-3,z],[0,1,0],7.47),cy(1.65,[x,-3.1,z],[0,1,0],7.67));sv('projector-clamp-spacer-'+name,spacer,'roll',m,'steel','OD5.6 ID3.3 length7.47; defines clip elevation.')
  sv('fastener-projector-'+name,bolt([x,-6,z],[0,1,0],20),'roll',m,'steel M3x20','Through clip + spacer + tray; washer and M3 nyloc above clip; head below tray to avoid floor contact.')
parts.remove(p);save('projector-open-tray',a,'roll',[0,0],'6061-T6 aluminium3','Open base tray with four peripheral clip tabs; exact fit checked against official TI geometry. Do not drill projector base.')
def readpart(name):
 p=next(p for p in parts if p['name']==name);r=STEPControl_Reader();r.ReadFile(str(out/(name+'.step')));r.TransferRoots();return p,r.OneShape()
for d in drive['motors']:
 role=d['role'];name='frame-'+role+'-motor-support';p,a=readpart(name);bp,b=readpart('fastener-'+role+'-rear-bridge');a=fuse(a,b);m=qm(d['quaternion'],d['tip'])
 for x in [14.5,21.5]:a=cut(a,move(countersink([x,-46.5,0],[0,1,0]),m))
 parts.remove(p);parts.remove(bp);save(name,a,p['parent'],[0,0],'fabricated steel rod6 and plate3','Joined metal bracket. Two flush M3x8 countersunk screws into motor saddle; no aluminium/steel welding interface.')
for side,xfrac in [('left',-.25),('right',.25)]:
 name='frame-optics-'+side;p,a=readpart(name);o=layout['mirror'];a=cut(a,move(countersink([xfrac*o['width'],-.18*o['height'],-14.1],[0,0,1]),dat['mirror']['matrix']));parts.remove(p);save(name,a,'roll',[0,0],p['material'],p['notes']+' Flush M3x12 mirror adjuster countersink.')
for key in ['mirror','foldMirror']:
 o=layout[key];m=dat[key]['matrix'];pts=[[-.25*o['width'],-.18*o['height']],[.25*o['width'],-.18*o['height']],[0,.25*o['height']]]
 for j,(x,y) in enumerate(pts):
  if key=='mirror' and j<2:continue
  a=cut(cyl(4.5,-14.1,3,x,y),countersink([x,y,-14.1],[0,0,1]));sv('fastener-'+key+'-seat-'+str(j),a,'roll',m,'steel spacer3','Countersunk 3mm stand-off supports adjustment screw head where no frame lug is present.')
# Folding mirror fixed plate has an independent central fastening to the optical frame.
p,a=readpart('foldMirror-adjustment-plate');a=cut(a,move(cyl(1.25,-11.2,3.2),dat['foldMirror']['matrix']));parts.remove(p);save(p['name'],a,'roll',[0,0],p['material'],p['notes']+' Central tapped M3 fixing, independent of three adjusters.')
p,a=readpart('frame-optics-left');a=cut(a,move(countersink([0,0,-14.1],[0,0,1]),dat['foldMirror']['matrix']));parts.remove(p);save(p['name'],a,'roll',[0,0],p['material'],p['notes']+' Folding mirror fixed by central M3x6.')
sv('fastener-foldMirror-frame',flatbolt([0,0,-14.1],[0,0,1],6),'roll',dat['foldMirror']['matrix'],'steel M3x6','Flush head through3mm steel lug into3mm tapped fixed mirror plate; independent of mirror adjustment.')
(out/'parts.json').write_text(json.dumps({'status':'v8 detailed prototype; no physical proof','parts':parts},indent=2));print('parts',len(parts))
