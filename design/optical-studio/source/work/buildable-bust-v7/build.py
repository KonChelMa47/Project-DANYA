# -*- coding: utf-8 -*-
import shutil
source=open('work/buildable-bust-v6/build.py').read();exec(source.split('# Horn bolts:')[0])
oldout=out;out=Path('outputs/danya-simulator/assets/buildable-bust-v7')
oldparts=json.loads((oldout/'parts.json').read_text())['parts'];parts=[]
replace={'yaw-ring-96T-retained','spring-guide'}
for p in oldparts:
 if p['name'] in replace or p['name'].startswith('carrier-standoff'):continue
 for ext in ['.step','.stl']:shutil.copy2(oldout/(p['name']+ext),out/(p['name']+ext))
 parts.append(p)
# Keep the tooth form; move carrier holes outward and provide two fasteners per leg.
a=gear(96,76,-24.5)
for y in [-41,41]:
 for x in [-12,12]:
  a=cut(a,cyl(1.65,-24.6,10.2,x,y));a=cut(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,-24.5),gp_Dir(0,0,1)),3.35,1.65,1.7).Shape())
save('yaw-ring-96T-retained',a,'yaw',[cx,cy],'PETG prototype','4xM3x20 countersunk; two per upper carrier leg at x+/-12 y+/-41.')
save('spring-guide',ring(2.6,1.65,-13.8,14.5),'pinion',[px,py],'POM','SHOWA AP080-022-1.2, OD8 ID5.6 free22, installed16-18. GuideOD5.2. UseM3x40 retaining bolt. Do not compress below16.')
for j,y in enumerate([-41,41]):
 for k,x in enumerate([-12,12]):save('carrier-standoff-'+str(j*2+k+1),ring(2.5,1.65,-14.5,6),'yaw',[cx+x,cy+y],'steel','L6 compression spacer. M3x20 through gear andspacer into tappedaluminiumfoot.')
# Upper yoke legs: two-bolt feet, machined bearing pockets and removable caps.
def box(x,y,z,dx,dy,dz):return BRepPrimAPI_MakeBox(gp_Pnt(x,y,z),dx,dy,dz).Shape()
def cyly(r,y,length,z=-4,x=0):return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,y,z),gp_Dir(0,1,0)),r,length).Shape()
for sy in [-1,1]:
 # Geometry created on positive side then mirrored by rotation about Z for the opposite leg.
 a=fuse(box(-17,39,-8.5,34,5.5,6),cyly(13,39,6));a=cut(a,cyly(4.2,38.9,6.2));a=cut(a,cyly(8.025,40,5.1))
 for x in [-12,12]:a=cut(a,cyl(1.25,-8.6,6.2,x,41))
 for angle in [90,210,330]:
  t=math.radians(angle);xx,zz=10*math.cos(t),-4+10*math.sin(t);a=cut(a,cyly(.8,41,4.1,zz,xx))
 a=cut(a,box(-30,30,-30,60,30,16))
 cap=cut(cyly(13,45,1.5),cyly(4.2,44.9,1.7));cap=cut(cap,box(-30,30,-30,60,30,16))
 for angle in [90,210,330]:
  t=math.radians(angle);xx,zz=10*math.cos(t),-4+10*math.sin(t);cap=cut(cap,cyly(1.1,44.9,1.7,zz,xx))
 if sy<0:
  from OCP.gp import gp_Trsf,gp_Ax1
  from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
  tr=gp_Trsf();tr.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)),math.pi);a=BRepBuilderAPI_Transform(a,tr,True).Shape();cap=BRepBuilderAPI_Transform(cap,tr,True).Shape()
 save('yaw-upper-leg-'+str(sy),a,'yaw',[cx,cy],'6061-T6 machined','688 bearing pocket16.05 prototype; production fit16H7 to be specified. M2x0.4 threads from1.6pilot holes. TwoM3x20 into tappedM3x0.5 feet; engagement4mm. PitchaxisZ-4; housing lowerflatZ-14.')
 save('pitch-bearing-cap-'+str(sy),cap,'yaw',[cx,cy],'6061-T6 aluminium1.5','3xM2x5 cap screws into leg, PCD20; throat8.4, bearing retained axially.')
# Standalone load-test interface plate and lever; these are not installed in the robot.
a=box(-60,-55,0,120,110,6)
for x in [-12,12]:
 for y in [-41,41]:a=cut(a,cyl(1.65,-.1,6.2,x,y))
for x in [-50,50]:
 for y in [-45,45]:a=cut(a,cyl(3.3,-.1,6.2,x,y))
for y in [-50,0]:a=cut(a,cyl(2.1,-.1,6.2,0,y))
a=cut(a,cyl(2.1,-.1,6.2,-50,0))
save('test-interface-plate',a,'fixture',[0,0],'6061-T6 aluminium6mm','Bench-only fixture. Four M6 bench holes and four yaw interface holes. Not for robot assembly.')
a=box(-10,-60,0,20,220,6)
for y in [-50,0,50,100,150]:a=cut(a,cyl(2.75,-.1,6.2,0,y))
save('test-load-lever',a,'fixture',[0,0],'6061-T6 aluminium6mm','Bench-only lever. Hole spacing50mm. Steel load attachment; not a human-support device.')
(out/'parts.json').write_text(json.dumps({'status':'v7 CAD candidate; full assembly interference and proof tests required','parts':parts},indent=2));print('VALID',len(parts))
