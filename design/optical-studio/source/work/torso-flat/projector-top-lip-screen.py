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
assembly=tool.GetShape_s(roots.Value(1))
for x in [65,120]:
 for z in [0,70]:
  # Top lip volume above nominal 3.97mm base thickness, 0.1mm assembly gap.
  box=BRepPrimAPI_MakeBox(gp_Pnt(x-3,4.07,z),6,3,5).Shape()
  op=BRepAlgoAPI_Common(box,assembly);op.Build();props=GProp_GProps();BRepGProp.VolumeProperties_s(op.Shape(),props)
  rows.append({'x':x,'z':z,'minCAD':[x-3,4.07,z],'sizeMM':[6,3,5],'booleanDone':op.IsDone(),'intersectionMM3':abs(props.Mass())});print(rows[-1],flush=True)
  Path('work/torso-flat/projector-top-lip-screen.json').write_text(json.dumps({'source':str(file),'scope':'Four candidate top lip boxes only; side and underside connection, motion, optics and retention strength not established. Nominal 0.1mm gap above base.','complete':len(rows)==4,'rows':rows},indent=2))
