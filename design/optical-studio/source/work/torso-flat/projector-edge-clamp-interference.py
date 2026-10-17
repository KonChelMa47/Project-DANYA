from pathlib import Path
import json
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString,TCollection_AsciiString
from OCP.XCAFDoc import XCAFDoc_DocumentTool,XCAFDoc_ShapeTool
from OCP.TDF import TDF_LabelSequence,TDF_Label,TDF_Tool
from OCP.TDataStd import TDataStd_Name
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopLoc import TopLoc_Location
p=Path('work/internal-projector').resolve()
file=next((p/'ti-cad').rglob('*TOP-ASSEMBLY_REVD.STEP'))
doc=TDocStd_Document(TCollection_ExtendedString('DANYA'));reader=STEPCAFControl_Reader();reader.SetNameMode(True);print('Read',reader.ReadFile(str(file)),flush=True);print('Transfer',reader.Transfer(doc),flush=True);tool=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main());roots=TDF_LabelSequence();tool.GetFreeShapes(roots);rows=[]
def name(label):
 a=TDataStd_Name()
 return a.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(),a) else ''
def entry(label):
 a=TCollection_AsciiString();TDF_Tool.Entry_s(label,a);return a.ToCString()



from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
assembly=tool.GetShape_s(roots.Value(1))

blocks=[([62,-9,62],[6,6,16.1]),([62,-9,75.1],[6,16.07,3]),([62,4.07,70],[6,3,8.1])]
for i,(origin,size) in enumerate(blocks):
 box=BRepPrimAPI_MakeBox(gp_Pnt(*origin),*size).Shape()
 op=BRepAlgoAPI_Common(box,assembly);op.Build();props=GProp_GProps();BRepGProp.VolumeProperties_s(op.Shape(),props)
 bounds=Bnd_Box();BRepBndLib.Add_s(op.Shape(),bounds)
 rows.append({'part':i,'origin':origin,'size':size,'booleanDone':op.IsDone(),'intersectionMM3':abs(props.Mass()),'intersectionBoundsCAD':None if bounds.IsVoid() else list(bounds.Get())});print(rows[-1],flush=True)
 Path('work/torso-flat/projector-edge-clamp-interference.json').write_text(json.dumps({'source':str(file),'complete':len(rows)==3,'rows':rows},indent=2))
