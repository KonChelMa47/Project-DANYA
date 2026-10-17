# Static optical bench. Units millimetres. Existing CAD helpers only; no installations.
exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/projection-prototype/assembly');out.mkdir(parents=True,exist_ok=True);parts=[]
I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
pm=dat['projector']['matrix'].copy();pm[12:15]=point([-111.62,-13.97,-81.33583308323333],pm);pm[14]+=2
fm=dat['foldMirror']['matrix'].copy();fm[14]+=2
baseholes=[]
def rod(p,q,r):
 d=[q[i]-p[i]for i in range(3)];return cy(r,p,d,math.sqrt(sum(t*t for t in d)))
def chain(pts,r):
 a=rod(pts[0],pts[1],r)
 for p,q in zip(pts[1:],pts[2:]):a=fuse(a,rod(p,q,r))
 for p in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*p),r).Shape())
 return a
def emit(n,a,note):save(n,a,'static',[0,0],'PETG',note)
def foot(x,y,spacing=12,width=40):
 a=box(x-width/2,y-10,-160,width,20,8)
 for xx in [x-spacing,x+spacing]:a=cut(a,cyl(2.2,-160.1,8.2,xx,y));baseholes.append([xx,y])
 return a
# Tall posts are single printable pieces. Upper arms bolt on at Z26.
for sg in [-1,1]:
 tag='left' if sg<0 else 'right';x=-90;y=sg*85
 a=fuse(foot(x,y,16,48),box(x-9,y-9,-153,18,18,179))
 a=fuse(a,box(x-24,y-12,18,48,24,8))
 for xx in [x-16,x+16]:a=cut(a,cyl(2.2,17.9,8.2,xx,y))
 columnShape=a
 emit('head-column-'+tag,a,'M4x25 base fasteners, M4x25 upper-arm fasteners; washers and nuts both ends. Print on broad side with supports.')
 a=box(x-24,y-12,26,48,24,8)
 a=fuse(a,rod([x,y,30],[-45,sg*38,30],7));a=fuse(a,rod([-45,sg*38,28],[-45,sg*38,78],5))
 for xx in [x-16,x+16]:a=cut(a,cyl(2.2,25.9,8.2,xx,y))
 for pad in json.load(open('work/buildable-bust-v8/head-pads.json')):
  if pad['point'][1]*sg<0:continue
  p=pad['point'].copy();p[2]+=8;n=pad['normal'];back=[p[i]-6*n[i]for i in range(3)]
  a=fuse(a,rod([-45,sg*38,p[2]],back,4));a=fuse(a,cy(5,back,n,2))
 a=cut(a,box(-500,-500,-500,1000,1000,526))
 emit('head-arm-'+tag,a,'Four conformal inner-shell contact feet added in Blender; bondline0.25mm. Bolt to column before bonding head to locate it.')
# Projector receiver held at its four existing nut pockets by independent stands.
for j,(xx,zz) in enumerate([(12,7),(12,68),(123,7),(123,68)]):
 q=point([xx,-23,zz],pm);end=point([xx,-12,zz],pm)
 a=fuse(foot(q[0],q[1]),chain([[q[0],q[1],-153],q,end],6))
 flange=move(cy(6,[xx,-14,zz],[0,1,0],6),pm);a=fuse(a,flange)
 # Clear access from rear for washer/head; no blind fastener trapped in support.
 a=cut(a,move(cy(1.7,[xx,-35,zz],[0,1,0],28),pm));a=cut(a,move(cy(3.7,[xx,-36,zz],[0,1,0],22),pm))
 rr=STEPControl_Reader();rr.ReadFile('outputs/danya-simulator/assets/buildable-bust-v10/drop-projector-tray-t5.step');rr.TransferRoots();mm=I.copy();mm[14]=8;a=cut(a,move(rr.OneShape(),mm))
 emit('projector-stand-'+str(j+1),a,'M3x10 plus0.5mm washer through6mm lug into tray captive M3 nut; M4x25 to base.')
# Fixed lower mirror, central rear entry, no motor clearance restrictions.
q=point([0,-14,-20.6],fm);end=point([0,0,-18.6],fm)
a=fuse(foot(-65,25),chain([[-65,25,-153],[-65,25,-75],q,end],5))
a=fuse(a,move(box(-9,-2.3,-21.6,18,4.6,6),fm))
for xx in [-5,5]:
 a=cut(a,move(cyl(1.7,-35,20,xx,0),fm));a=cut(a,move(cyl(3.7,-40,18.4,xx,0),fm))
w=layout['foldMirror']['width'];h=layout['foldMirror']['height']
for xx,yy in [(-.25*w,-.18*h),(.25*w,-.18*h),(0,.25*h)]:a=cut(a,move(cyl(3.7,-19.3,3.8,xx,yy),fm))
a=cut(a,move(box(-500,-500,-15.6,1000,1000,1000),fm))
emit('lower-mirror-stand',a,'Two M3x12 screws with washers through6mm flange, into mirror fixed-plate captive nuts. M4x25 base fasteners.')
# Split open plinth, 240x220mm assembled. Each half120x220, AD5M fit only without brim; A1 preferred.
for sg in [-1,1]:
 x0=-120 if sg<0 else 0;a=box(x0,-110,-170,120,220,10)
 for x,y in baseholes:
  if x0-2<x<x0+122:a=cut(a,cyl(2.2,-170.1,10.2,x,y))
 # Joining straps are below base; four points per strap.
 for x in [-10,10]:
  if x0<=x<=x0+120:
   for y in [-50,65]:a=cut(a,cyl(2.2,-170.1,10.2,x,y))
 emit('base-'+('left'if sg<0 else'right'),a,'M4x25 mount screws. Two underside joining straps; use included feet so straps clear table.')
for i,y in enumerate([-50,65]):
 a=box(-22,y-10,-176,44,20,6)
 for x in [-10,10]:a=cut(a,cyl(2.2,-176.1,6.2,x,y))
 emit('base-join-'+str(i+1),a,'Two M4x25 bolts, washers and nuts; clamp both base halves.')
for i,(x,y)in enumerate([(-100,-101),(-100,101),(100,-101),(100,101)]):
 a=cut(cyl(9,-182,12,x,y),cyl(2.2,-182.1,12.2,x,y));a=cut(a,cyl(5,-182.1,5.1,x,y));emit('base-foot-'+str(i+1),a,'M4x25 screw through foot and base; washer and nut. Add adhesive rubber sole.')
 # add to relevant base in later pass below
for p in list(parts):
 if p['name']in ['base-left','base-right']:
  rr=STEPControl_Reader();rr.ReadFile(str(out/(p['name']+'.step')));rr.TransferRoots();a=rr.OneShape()
  for x,y in [(-100,-101),(-100,101),(100,-101),(100,101)]:
   if (x<0)==(p['name']=='base-left'):a=cut(a,cyl(2.2,-170.1,10.2,x,y))
  old=p.copy();parts.remove(p);emit(old['name'],a,old['notes'])
# Exact nominal cutting masters: outside edge is glass edge, not holder clearance.
templates=[]
for key in ['mirror','foldMirror']:
 o=layout[key];pts=[(x*o['width'],y*o['height'])for x,y in o.get('outline',[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])]
 a=poly(pts,0,2);a=cut(a,poly([(-1,-3),(1,-3),(1,1),(3,1),(0,5),(-3,1),(-1,1)],1.3,.8));emit(key+'-cutting-master',a,'Print100% scale. Trace OUTSIDE edge onto mirror back; no kerf compensation. Not an optical mirror. Nominal glass1.1mm.')
 templates.append({'name':key,'widthMM':o['width'],'heightMM':o['height'],'glassThicknessAssumedMM':1.1,'pointsMM':pts})
 # Dimensioned1:1 SVG: physical units explicit, no printer fit-to-page.
 W=o['width'];H=o['height'];points=' '.join(f'{x+W/2+10:.4f},{H/2-y+10:.4f}'for x,y in pts)
 (out.parent/(key+'-cutting-outline.svg')).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W+20}mm" height="{H+35}mm" viewBox="0 0 {W+20} {H+35}"><polygon points="{points}" fill="none" stroke="black" stroke-width="0.15"/><text x="10" y="{H+18}" font-size="3">{key}: {W:.3f} x {H:.3f} mm | 1:1</text><path d="M10,{H+25} h20" stroke="black"/><text x="10" y="{H+30}" font-size="3">20 mm check</text></svg>')
(out.parent/'generated-parts.json').write_text(json.dumps({'parts':parts,'baseHoles':baseholes,'projectorBaseMatrix':pm,'mirrorTemplates':templates},indent=2))
print('GENERATED',len(parts))
