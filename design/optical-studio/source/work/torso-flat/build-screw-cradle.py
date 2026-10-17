import json,math
from pathlib import Path
from OCP.gp import gp_Pnt,gp_Dir,gp_Ax2
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Cut,BRepAlgoAPI_Common
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
w=Path('work/torso-flat');out=w/'screw-cradle';out.mkdir(exist_ok=True)
def cylinder(a,b,r):
 d=[b[i]-a[i] for i in range(3)];return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*a),gp_Dir(*d)),r,math.sqrt(sum(v*v for v in d))).Shape()
def boolean(cls,a,b):
 op=cls(a,b);op.Build()
 if not op.IsDone():raise RuntimeError('Boolean failed')
 return op.Shape()
def volume(s):
 p=GProp_GProps();BRepGProp.VolumeProperties_s(s,p);return abs(p.Mass())
parts=[cylinder(r['a'],r['b'],r['radius']) for r in json.loads((w/'projector-cradle-cad-lines.json').read_text())]
for x in [13.742,61.742]:parts.append(cylinder([x,-10,48.0002],[x,0,48.0002],4))
s=parts[0]
for p in parts[1:]:s=boolean(BRepAlgoAPI_Fuse,s,p)
# Drill AFTER union so connector material cannot refill the fastener passages.
for x in [13.742,61.742]:
 s=boolean(BRepAlgoAPI_Cut,s,cylinder([x,-11,48.0002],[x,.1,48.0002],1.2))
 s=boolean(BRepAlgoAPI_Cut,s,cylinder([x,-11,48.0002],[x,-8,48.0002],2.2))
valid=BRepCheck_Analyzer(s).IsValid();ex=TopExp_Explorer(s,TopAbs_SOLID);count=0
while ex.More():count+=1;ex.Next()
access=[]
for x in [13.742,61.742]:
 tool=cylinder([x,-60,48.0002],[x,-8.1,48.0002],2)
 access.append({'axisCAD':[x,48.0002],'shaftDiameterMM':4,'fromY':-60,'toY':-8.1,'frameIntersectionMM3':volume(boolean(BRepAlgoAPI_Common,s,tool))})
writer=STEPControl_Writer();writer.Transfer(s,STEPControl_AsIs);writer.Write(str(out/'cradle.step'))
BRepMesh_IncrementalMesh(s,.1,False,.2,True).Perform();StlAPI_Writer().Write(s,str(out/'cradle.stl'))
report={'scope':'Geometric prototype in official projector CAD coordinates. Original occupied screws require replacement with +10mm under-head length. Strength, material, threads, external assembly access and final mounting not qualified. Tool test is detached cradle only, 4mm shaft assumed.','valid':valid,'solidCount':count,'volumeMM3':volume(s),'toolAccess':access}
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
