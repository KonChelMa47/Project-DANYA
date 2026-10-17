exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.gp import gp_Ax1
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
out=Path('outputs/danya-simulator/assets/projection-prototype/tray-nut-retention-v4');parts=[]
pm=json.load(open('outputs/danya-simulator/assets/projection-prototype/generated-parts.json'))['projectorBaseMatrix']
# Current v10 source has -6mm optical drop, static assembly then adds8mm globally.
r=STEPControl_Reader();assert r.ReadFile('outputs/danya-simulator/assets/buildable-bust-v10/drop-projector-tray-t5.step')==1;r.TransferRoots();tray=r.OneShape();lift=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,8,1];tray=move(tray,lift)
I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
def localshape(shape,x,z,sg):
 # Flip X/Z together for the upper pair, so a single cap fits all four seats.
 M=[sg,0,0,0,0,1,0,0,0,0,sg,0,x,0,z,1];return move(move(shape,M),pm)
def extrude_xz(points,y,h):
 # poly starts inXY withZ extrusion; rotate so XY->XZ and extrusion->+Y.
 a=poly([(x,-z)for x,z in points],0,h);t=gp_Trsf();t.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(1,0,0)),-math.pi/2);a=BRepBuilderAPI_Transform(a,t,True).Shape();return move(a,[1,0,0,0,0,1,0,0,0,0,1,0,0,y,0,1])
# Sliding T-rails carry axial retention; two flexible side detents block withdrawal.
cap=box(-5.5,-8,-5,11,.8,11.3)
for sg in [-1,1]:
 x=sg*4.7;cap=fuse(cap,box(x-.4,-7.25,-4.6,.8,.85,9.2));cap=fuse(cap,box(x-.75,-6.65,-4.6,1.5,.35,9.2))
 # Free each outside rail over8.0mm. Back1.0mm remains the spring root.
 cap=cut(cap,box(3.2 if sg>0 else-3.9,-8.1,-3.7,.7,2.2,10.2))
 pts=[(sg*5.45,2.5),(sg*6.05,3.35),(sg*6.05,4.35),(sg*5.45,4.35)]
 cap=fuse(cap,extrude_xz(pts,-8,.8))
cap=cut(cap,cy(1.9,[0,-8.1,0],[0,1,0],2.5))
def receiver():
 a=box(-7.3,-8,-6,14.6,5,13)
 # Nut chamber and blind screw bore, independent of cap channels.
 a=cut(a,hex_y(0,-7.05,0,2.55,5.8));a=cut(a,cy(1.7,[0,-8.1,0],[0,1,0],4.0))
 # Open inner edge permits slide-in assembly; external underside plane remainsY=-8.
 a=cut(a,box(-5.7,-8.1,-5.2,11.4,1.1,12.4))
 for sg in [-1,1]:
  x=sg*4.7;a=cut(a,box(x-.55,-7.15,-4.8,1.1,.7,12));a=cut(a,box(x-.95,-6.8,-4.8,1.9,.7,12))
  a=cut(a,box(5.65 if sg>0 else-6.3,-8.1,2.7,.65,1.1,1.95))
 return a
seat=receiver()
# Fill old underside pockets locally, then cut the exact new receiver voids into filled tray.
voids=cut(box(-7.3,-8,-6,14.6,5,13),seat)
for i,(x,z,sg)in enumerate([(12,7,1),(12,68,-1),(123,7,1),(123,68,-1)]):
 tray=fuse(tray,localshape(box(-7.3,-8,-6,14.6,5,13),x,z,sg));tray=cut(tray,localshape(voids,x,z,sg));tray=cut(tray,localshape(box(-6.35,-8.1,6.9,12.7,2.1,14.1),x,z,sg))
 sv('tray-nut-cover-'+str(i),move(cap,[sg,0,0,0,0,1,0,0,0,0,sg,0,x,0,z,1]),'static',pm,'printed PETG','Slide from ventilation window. T-rails prevent axial dropout; paired side detents stop withdrawal. Physical fit test required. 4identical covers.')
 nut=cut(hex_y(x,-6.95,z,2.4,5.5),cy(1.3,[x,-7.05,z],[0,1,0],2.6));sv('drop-projector-tray-t5-frame-nut-'+str(i),nut,'static',pm,'purchased M3 nut AF5.5 t2.4','Retained by sliding cover before stand installation. Shifted0.95mm deeper; existing M3x10 unchanged.')
save('drop-projector-tray-t5',tray,'static',[0,0],'printed PETG','Static-only v4: four slide-locked nut covers. Base socket, projector datums, screw axes and stand mating plane retained; physical/thermal qualification pending.')
save('tray-nut-fit-coupon',seat,'fixture',[0,0],'printed PETG','Print one seat and one cover first; same clearances as the tray. Not assembled into robot.')
save('tray-nut-cover-local',cap,'fixture',[0,0],'printed PETG','Canonical cap for print orientation and fit checks; use four assembled coverfiles.')
# Exact CAD overlap and sliding rail path checks, excluding intended elastic detent interference during insertion.
def volume(a):
 prop=GProp_GProps();BRepGProp.VolumeProperties_s(a,prop);return abs(prop.Mass())
checks={'coverSeatOverlapMM3':volume(BRepAlgoAPI_Common(cap,seat).Shape()),'nutSeatOverlapMM3':volume(BRepAlgoAPI_Common(cut(hex_y(0,-6.95,0,2.4,5.5),cy(1.3,[0,-7.05,0],[0,1,0],2.6)),seat).Shape()),'coverThicknessMM':.8,'coverScrewClearanceDiameterMM':3.8,'nutPocketAFMM':5.8,'nutYRangeMM':[-6.95,-4.55],'standMatingPlaneLocalYMM':-8,'existingScrewTipLocalYMM':-4.5,'blindBoreEndLocalYMM':-4.1,'detentDeflectionNominalMM':.35,'mechanism':'Sliding T rails, 0.2mm side clearance, two side detents; nut chamber under removable cap','physicalInsertionPullTest':False}
(out/'cad-check.json').write_text(json.dumps(checks,indent=2));(out/'parts.json').write_text(json.dumps({'parts':parts,'matrix':pm,'checks':checks},indent=2));print('CHECKS',checks);assert checks['coverSeatOverlapMM3']<1e-5 and checks['nutSeatOverlapMM3']<1e-5
