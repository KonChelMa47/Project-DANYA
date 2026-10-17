import json,math
from pathlib import Path
from OCP.gp import gp_Pnt,gp_Dir,gp_Ax2
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder,BRepPrimAPI_MakeCone,BRepPrimAPI_MakePrism,BRepPrimAPI_MakeBox
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon,BRepBuilderAPI_MakeFace
from OCP.gp import gp_Vec
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Cut
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
out=Path('outputs/danya-simulator/assets/buildable-bust-v6');out.mkdir(exist_ok=True)
g=json.load(open('outputs/danya-simulator/assets/buildable-bust-v5/gear-layout.json'));cx,cy,_=g['center'];px,py,_=g['pinionCenter'];dx,dy=px-cx,py-cy
parts=[]
def cyl(r,z,h,x=0,y=0):return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,y,z),gp_Dir(0,0,1)),r,h).Shape()
def cut(a,b):return BRepAlgoAPI_Cut(a,b).Shape()
def fuse(a,b):return BRepAlgoAPI_Fuse(a,b).Shape()
def ring(ro,ri,z,h):return cut(cyl(ro,z,h),cyl(ri,z-.1,h+.2))
def poly(points,z,h):
 p=BRepBuilderAPI_MakePolygon()
 last=None
 for x,y in points:
  if last and math.hypot(x-last[0],y-last[1])<1e-7:continue
  p.Add(gp_Pnt(x,y,z));last=(x,y)
 p.Close();return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(p.Wire()).Face(),gp_Vec(0,0,h)).Shape()
def hexagon(af,z,h):return poly([(af/math.sqrt(3)*math.cos(i*math.pi/3),af/math.sqrt(3)*math.sin(i*math.pi/3))for i in range(6)],z,h)
def save(name,s,parent,xy,material,notes):
 assert BRepCheck_Analyzer(s).IsValid(),name
 e=TopExp_Explorer(s,TopAbs_SOLID);n=0
 while e.More():n+=1;e.Next()
 assert n==1,(name,n)
 BRepMesh_IncrementalMesh(s,.045,False,.12,True).Perform();w=StlAPI_Writer();w.ASCIIMode=False;w.Write(s,str(out/(name+'.stl')))
 w=STEPControl_Writer();w.Transfer(s,STEPControl_AsIs);w.Write(str(out/(name+'.step')))
 parts.append(dict(name=name,parent=parent,xy=xy,material=material,notes=notes,validSolid=True,solidCount=n))
def gear(teeth,bore,z):
 rp=teeth/2;rb=rp*math.cos(math.radians(20));ra=rp+1;rf=rp-1.25;half=math.pi/(2*teeth)-.1/(2*rp);ip=math.tan(math.radians(20))-math.radians(20)
 def ang(r):a=math.acos(rb/r);return half+ip-(math.tan(a)-a)
 pts=[]
 for k in range(teeth):
  a=k*2*math.pi/teeth;start=max(rb,rf)
  def add(r,t):pts.append((r*math.cos(a+t),r*math.sin(a+t)))
  add(rf,-math.pi/teeth);add(rf,-ang(start))
  if rb>rf:add(rb,-ang(rb))
  for j in range(13):r=start+(ra-start)*j/12;add(r,-ang(r))
  for j in range(1,5):add(ra,-ang(ra)+2*ang(ra)*j/4)
  for j in range(11,-1,-1):r=start+(ra-start)*j/12;add(r,ang(r))
  if rb>rf:add(rf,ang(rb))
  add(rf,math.pi/teeth)
 return cut(poly(pts,z,10),cyl(bore/2,z-.1,10.2))
# Horn bolts: four holes on radius7 confirmed CAD. M3 thread specification remains an inspection gate.
a=fuse(cyl(13,-30.4,5),cyl(4,-25.4,9.9));a=cut(a,cyl(1.65,-31,20));a=cut(a,hexagon(5.7,-30.5,3.3))
for x,y in [(7,0),(-7,0),(0,7),(0,-7)]:
 a=cut(a,cyl(1.65,-30.5,5.2,x,y));a=cut(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,-27.1),gp_Dir(0,0,1)),1.65,3.35,1.7).Shape())
save('horn-clutch-hub',a,'pinion',[px,py],'PETG prototype','4xM3x8 countersunk screws; radius7. Captive M3x30 hex bolt; retained original servo horn/spline screw. Smooth journal for slipping pinion, NOT a key.')
save('pinion-20T-clutch',gear(20,8.2,-25),'pinion',[px,py],'PETG prototype','10mm face, 8.2mm sliding bore, axial spring retention. Torque must be calibrated.')
a=gear(96,76,-24.5)
for y in [-42,42]:
 a=cut(a,cyl(1.65,-24.6,10.2,0,y));a=cut(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(0,y,-24.5),gp_Dir(0,0,1)),3.35,1.65,1.7).Shape())
save('yaw-ring-96T-retained',a,'yaw',[cx,cy],'PETG prototype','10mm face.2xM3x30 countersunk load-transfer bolts at radius42.Optical bore76mm.')
# Fixed captured annulus. Notch clears servo adapter and cap at pinion axis.
a=ring(46,36,-28,3)
for i in range(6):
 t=i*math.pi/3;a=fuse(a,cyl(8,-28,3,50*math.cos(t),50*math.sin(t)))
a=cut(a,cyl(15,-32,8,dx,dy))
for i in range(6):
 t=i*math.pi/3;x,y=54*math.cos(t),54*math.sin(t);a=cut(a,cyl(1.65,-32,8,x,y));a=cut(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,-28),gp_Dir(0,0,1)),3.35,1.65,1.7).Shape())
save('yaw-bearing-base',a,'fixed',[cx,cy],'6061-T6 aluminium 3mm','Six M3 through bolts on PCD108; frame attachment via same bolt stack. Shaft radial bushing separate.')
save('yaw-radial-sleeve',ring(37.8,36,-25,10.8),'fixed',[cx,cy],'POM or machined low-friction polymer','0.2mm nominal radial clearance. OD75.6 ID72. Captured by base and upper plate.')
a=ring(46,38,-25,.5);a=cut(a,cyl(15,-26,2,dx,dy));save('lower-thrust-liner',a,'fixed',[cx,cy],'PTFE sheet','0.5mm thickness; supported thrust face.')
a=ring(58,46,-13.7,3);a=cut(a,cyl(8,-14,4,dx,dy))
for i in range(6):t=i*math.pi/3;a=cut(a,cyl(1.65,-14,4,54*math.cos(t),54*math.sin(t)))
save('upper-retaining-plate',a,'fixed',[cx,cy],'6061-T6 aluminium 3mm','Do not substitute printed plastic without recalculation. SixM3x25 bolts, spacers11.3mm.')
a=ring(49,46,-14.2,.5);save('upper-thrust-liner',a,'fixed',[cx,cy],'PTFE sheet','0.3mm nominal upward travel before liner contact.')
for name,z in [('clutch-lower-friction',-25.4),('clutch-upper-friction',-15)]:save(name,ring(8.5,4.1,z,.4),'pinion',[px,py],'friction sheet','0.4mm; coefficient not assumed for release torque, calibrate 0.08 +/-0.01Nm; reject above0.10Nm.')
save('clutch-pressure-washer',ring(8.5,1.65,-14.6,.8),'pinion',[px,py],'steel shim','OD17 ID3.3 thickness0.8; custom washer.')
save('spring-guide',ring(2.8,1.65,-13.8,6.5),'pinion',[px,py],'PETG prototype','Loose guide, must not bottom out; compression spring free16mm OD8 ID6, target installed8mm; spring rate selection/calibration required.')
for i in range(6):
 t=i*math.pi/3;save('retainer-spacer-'+str(i+1),ring(4,1.65,-25,11.3),'fixed',[cx+54*math.cos(t),cy+54*math.sin(t)],'steel or aluminium','OD8 ID3.3 length11.3; M3x25 through stack, washer+nyloc below base.')
for i,sy in enumerate([-1,1]):save('carrier-standoff-'+str(i+1),ring(3,1.65,-14.5,6),'yaw',[cx,cy+sy*42],'steel or aluminium','Output attachment spacer. Foot bridging to existing yoke must be detailed in whole-frame assembly.')
(out/'parts.json').write_text(json.dumps({'status':'Detailed prototype solids; bench proof and actual horn threads required before release. Not a certified robot assembly.', 'parts':parts},indent=2));print('VALID SOLIDS',len(parts))
