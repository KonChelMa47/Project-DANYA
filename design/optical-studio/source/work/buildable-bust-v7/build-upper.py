# -*- coding: utf-8 -*-
import json,math
from pathlib import Path
source=open('work/buildable-bust-v6/build.py').read();exec(source.split('# Horn bolts:')[0]);out=Path('outputs/danya-simulator/assets/buildable-bust-v7');parts=json.loads((out/'parts.json').read_text())['parts'];parts=[p for p in parts if not p['name'].startswith(('pitch-yoke-v7','roll-carrier-v7','axis-bearing','axis-shaft','roll-bearing-cap','clutch-test','test-base-adapter'))]
from OCP.gp import gp_Trsf,gp_Ax1
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

def box(x,y,z,dx,dy,dz):return BRepPrimAPI_MakeBox(gp_Pnt(x,y,z),dx,dy,dz).Shape()
def cyly(r,y,length,z=20,x=0):return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,y,z),gp_Dir(0,1,0)),r,length).Shape()
def cylx(r,x,length,z=20,y=0):return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,y,z),gp_Dir(1,0,0)),r,length).Shape()
def mirror(s):
 t=gp_Trsf();t.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)),math.pi);return BRepBuilderAPI_Transform(s,t,True).Shape()
a=ring(50,46,17,6);a=cut(a,box(-55,38.5,16,110,20,8));a=cut(a,box(-55,-58.5,16,110,20,8))
for sy in [-1,1]:
 bridge=box(-32,34.5,17,64,4,6);hub=cyly(7,34.5,4,-4);bridge=fuse(bridge,box(-7,34.5,-4,14,4,24));bridge=fuse(bridge,hub);bridge=cut(bridge,cyly(4.025,34.4,4.2,-4));bridge=cut(bridge,cylx(1.5,-8,16,-4,36.5));a=fuse(a,bridge if sy>0 else mirror(bridge))
 # Bearing housing for roll, bearing inserted from outer side.
 housing=cylx(13,40,6);housing=cut(housing,cylx(4.2,39.9,6.2));housing=cut(housing,cylx(8.025,41,5.1))
 for deg in [90,210,330]:
  t=math.radians(deg);y,z=10*math.cos(t),20+10*math.sin(t);housing=cut(housing,cylx(.8,42,4.1,z,y))
 a=fuse(a,housing if sy>0 else mirror(housing))
 cap=cut(cylx(13,46,1.5),cylx(4.2,45.9,1.7))
 for deg in [90,210,330]:
  t=math.radians(deg);y,z=10*math.cos(t),20+10*math.sin(t);cap=cut(cap,cylx(1.1,45.9,1.7,z,y))
 save('roll-bearing-cap-'+str(sy),cap if sy>0 else mirror(cap),'pitch',[cx,cy],'6061-T6 aluminium1.5','ThreeM2x5 into tapped1.6pilot holes; 688 bearing end retention.')
# Bypass the roll belt with a bridge between its two runs; no change to outer neck profile.
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
a=cut(a,box(31,20,16,9,24,8))
bypass=poly([(28,34),(30,39),(43,29),(41,24)],5,4)
bypass=fuse(bypass,box(28,34,5,3,4.5,18));bypass=fuse(bypass,box(40,22,5,3,7,18));bypass=BRepAlgoAPI_Common(bypass,cyl(50,4,20)).Shape();a=fuse(a,bypass)
save('pitch-yoke-v7',a,'pitch',[cx,cy],'6061-T6 machined','Pitch bridge and roll bearing housings. Steel8mm shaft cross-pinnedM3 at Y+/-36.5. PitchaxisZ-4; rollaxisZ20.')
# Raise the inner annulus without moving the head or its rotation centre.
a=ring(43,39,37,6)
for sy in [-1,1]:
 if sy>0:
  arm=box(25,-6,20,4,12,17);arm=fuse(arm,cylx(7,25,4));arm=fuse(arm,box(25,-6,35,16,12,6));arm=cut(arm,cylx(4.025,24.9,4.2));arm=cut(arm,cyly(1.5,-8,16,20,27))
 else:
  arm=box(-38,-6,20,4,12,17);arm=fuse(arm,cylx(7,-38,4));arm=fuse(arm,box(-41,-6,35,5,12,6));arm=cut(arm,cylx(4.025,-38.1,4.2));arm=cut(arm,cyly(1.5,-8,16,20,-36))
 a=fuse(a,arm)
save('roll-carrier-v7',a,'roll',[cx,cy],'6061-T6 machined','Raised annuluscenterZ40, unchanged optical/head pivotZ20. Asymmetric arms clear inboard roll pulley. Existing head/optics mounting beyond annulus remains separate.')
for sy in [-1,1]:
 bearing=cut(cyly(8,40,5,-4),cyly(4,39.9,5.2,-4));save('axis-bearing-pitch-'+str(sy),bearing if sy>0 else mirror(bearing),'yaw',[cx,cy],'688ZZ purchased','8x16x5 envelope; not a printable bearing.')
 bearing=cut(cylx(8,41,5),cylx(4,40.9,5.2));save('axis-bearing-roll-'+str(sy),bearing if sy>0 else mirror(bearing),'pitch',[cx,cy],'688ZZ purchased','8x16x5 envelope; not a printable bearing.')
 shaft=cyly(4,34.5,40.5 if sy>0 else 15,-4);shaft=cut(shaft,cylx(1.5,-5,10,-4,36.5));save('axis-shaft-pitch-'+str(sy),shaft if sy>0 else mirror(shaft),'pitch',[cx,cy],'steel8h6','Crosshole3.0; axial M3 tapped end for end washer, thread not modelled; pulley cross-pin/hub interface requires matching purchased pulley.')
 x0=25 if sy>0 else -49.5;L=24.5 if sy>0 else 15.5
 shaft=cylx(4,x0,L);shaft=cut(shaft,cyly(1.5,-5,10,20,27 if sy>0 else -36));save('axis-shaft-roll-'+str(sy),shaft,'roll',[cx,cy],'steel8h6','Crosshole3.0 into carrier arm; axial M3 end-retaining screw thread not modelled.')
# Bench fixtures: independent from installed assembly.
a=box(-90,-80,0,180,160,8)
for i in range(6):
 t=i*math.pi/3;a=cut(a,cyl(1.65,-.1,8.2,54*math.cos(t),54*math.sin(t)))
for x in [-75,75]:
 for y in [-65,65]:a=cut(a,cyl(3.3,-.1,8.2,x,y))
save('test-base-adapter',a,'fixture',[0,0],'6061-T6 aluminium8','PCD108 sixholes3.3; fourM6benchholes. Test-only M3x35 stack, fixture rests rigidbench.')
a=ring(17,11.1,0,8);a=fuse(a,box(12,-5,0,44,10,8));a=fuse(a,box(-20,-8,0,8,16,8));a=cut(a,box(-21,-.5,-.1,22,1,8.2));a=cut(a,cyl(2.2,-.1,8.2,50,0));a=cut(a,BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-16,-8.1,4),gp_Dir(0,1,0)),1.65,16.2).Shape())
save('clutch-test-lever',a,'fixture',[0,0],'PETG test fixture','Splitclamp22.2for22mm pinionOD; M3clampbolt. Forceholeat50mm; calibrate torque1.4-1.8N without disturbing axial clutch stack.')
(out/'parts.json').write_text(json.dumps({'status':'Prototype detail; physical proof not performed','parts':parts},indent=2));print('parts',len(parts))
