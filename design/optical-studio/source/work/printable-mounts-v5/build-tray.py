exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
out=Path('outputs/danya-simulator/assets/projection-prototype/printable-mounts-v5');parts=[]
pm=json.load(open(out.parent/'generated-parts.json'))['projectorBaseMatrix']
def volume(a):
 p=GProp_GProps();BRepGProp.VolumeProperties_s(a,p);return abs(p.Mass())
def xyzpoly(points,z,h):return poly(points,z,h)
def place(a,x,z,sg):return move(move(a,[sg,0,0,0,0,1,0,0,0,0,sg,0,x,0,z,1]),pm)
# Bottom-up receiver: all nut-pocket walls grow directly from a flat floor.
seat=box(-7.3,-8,-6,14.6,5,13)
seat=cut(seat,hex_y(0,-5.65,0,2.8,5.8));seat=cut(seat,cy(1.7,[0,-8.1,0],[0,1,0],5.3))
for sg in [-1,1]:
 pts=[(sg*x,y)for x,y in [(5.05,-3),(7.3,-3),(7.3,-1.4),(3.45,-1.4)]]
 seat=fuse(seat,xyzpoly(pts,-5.2,11.8))
 # Latch recess remains open to the top: no bridging over the nut pocket.
 seat=cut(seat,box(4.15 if sg>0 else-5.65,-2.9,2.65,1.5,2,2.1))
seat=fuse(seat,box(-7.3,-3,-6,14.6,1.6,.8))
# Removable dovetail support shoe replaces the original small round datum pad.
# Full flat underside is the print bed; sloped sides narrow upwards.
cap=xyzpoly([(-4.75,-2.8),(4.75,-2.8),(3.15,-1.2),(3.15,0),(-3.15,0),(-3.15,-1.2)],-5,11.3)
cap=cut(cap,cy(1.9,[0,-2.9,0],[0,1,0],1.5))
for sg in [-1,1]:
 cap=cut(cap,box(2.2 if sg>0 else-2.9,-2.9,-3.7,.7,3.1,10.2))
 # Spring noses below the datum surface. Short sloped entry ramp in Z.
 pts=[(sg*x,y)for x,y in [(4.7,-2.8),(5.15,-2.8),(4.6,-2.25),(4.15,-2.25)]]
 hook=xyzpoly(pts,3.0,1.4);cap=fuse(cap,hook)
# The spring arms do not support the base; central datum at Y=0 stays rigid.
nut=cut(hex_y(0,-5.55,0,2.4,5.5),cy(1.3,[0,-5.65,0],[0,1,0],2.6))
r=STEPControl_Reader();assert r.ReadFile('outputs/danya-simulator/assets/buildable-bust-v10/drop-projector-tray-t5.step')==1;r.TransferRoots();tray=move(r.OneShape(),[1,0,0,0,0,1,0,0,0,0,1,0,0,0,8,1])
checks=[]
for i,(x,z,sg)in enumerate([(12,7,1),(12,68,-1),(123,7,1),(123,68,-1)]):
 # Replace complete old blind underside pocket and round support pad.
 tray=cut(tray,place(box(-7.29,-8.1,-5.99,14.58,8.3,12.98),x,z,sg))
 tray=fuse(tray,place(seat,x,z,sg))
 sv('tray-nut-cover-'+str(i),move(cap,[sg,0,0,0,0,1,0,0,0,0,sg,0,x,0,z,1]),'static',pm,'printed PETG','v5 top-loaded nut; dovetail shoe slides from vent window, central datumY0. Print flat underside; test spring engagement.')
 sv('drop-projector-tray-t5-frame-nut-'+str(i),move(nut,[1,0,0,0,0,1,0,0,0,0,1,0,x,0,z,1]),'static',pm,'M3 hex nut AF5.5 t2.4','Top-loaded nut. Frame screw now M3x12.')
 # Existing stand and washer stack: under-head Y=-14.5, screw tip=-2.5.
 sv(f'static-hardware-projector-{x}-{z}-bolt',fuse(cy(1.5,[x,-14.5,z],[0,1,0],12),cy(2.75,[x,-17.5,z],[0,1,0],3)),'static',pm,'purchased M3x12','Replacement for tray-to-stand M3x10; confirm actual stack before tightening.')
save('drop-projector-tray-t5',tray,'static',[0,0],'printed PETG','v5 nut floors bed-supported, hex chambers open upward, removable datum shoes. Original projector datum and socket retained.')
save('tray-nut-fit-coupon',seat,'fixture',[0,0],'printed PETG','Print floorY-8 downward, hex pocket up; dovetail walls are45degree slopes. No nut-pocket support.')
save('tray-nut-cover-local',cap,'fixture',[0,0],'printed PETG','Print flatY-2.8 face down. Four identical shoes.')
report={'capSeatOverlapMM3':volume(BRepAlgoAPI_Common(cap,seat).Shape()),'nutSeatOverlapMM3':volume(BRepAlgoAPI_Common(nut,seat).Shape()),'nutCapOverlapMM3':volume(BRepAlgoAPI_Common(nut,cap).Shape()),'nutPocketOpenDirection':'+Y (print up)','nutPocketFloorYMM':-5.65,'bedYMM':-8,'floorThicknessMM':2.35,'nutYRangeMM':[-5.55,-3.15],'supportDatumYMM':0,'frameScrew':'M3x12','frameScrewTipYMM':-2.5,'coverBlindBoreEndYMM':-1.4,'physicalTests':False}
(out/'tray-cad-check.json').write_text(json.dumps(report,indent=2));(out/'tray-parts.json').write_text(json.dumps({'parts':parts,'matrix':pm},indent=2));print(report)
assert max(report[k]for k in ['capSeatOverlapMM3','nutSeatOverlapMM3','nutCapOverlapMM3'])<1e-5
