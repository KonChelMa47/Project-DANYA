import json
from pathlib import Path
from OCP.gp import gp_Pnt,gp_Dir,gp_Ax2
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox,BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Cut
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
out=Path('outputs/danya-simulator/assets/buildable-bust-v2');reports=[]
for d in [3.2,3.4,3.6]:
 a=BRepPrimAPI_MakeBox(20,24,3).Shape();b=BRepPrimAPI_MakeBox(6,24,18).Shape();s=BRepAlgoAPI_Fuse(a,b).Shape();hole=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-1,12,10),gp_Dir(1,0,0)),d/2,8).Shape();s=BRepAlgoAPI_Cut(s,hole).Shape();ok=BRepCheck_Analyzer(s).IsValid();assert ok
 name=f'M3-joint-coupon-hole-{d:.1f}';BRepMesh_IncrementalMesh(s,.05,False,.15,True).Perform();w=StlAPI_Writer();w.Write(s,str(out/f'{name}.stl'));w=STEPControl_Writer();w.Transfer(s,STEPControl_AsIs);w.Write(str(out/f'{name}.step'));reports.append({'part':name,'sizeMM':[20,24,18],'validSolid':ok,'printQuantityForPair':2,'targetFastener':'M3x20 + 2 washers + M3 nut; 0.4 mm spacer during fit trial'})
(out/'coupon-audit.json').write_text(json.dumps(reports,indent=2))
