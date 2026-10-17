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


from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt,gp_Ax2,gp_Dir
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
assembly=tool.GetShape_s(roots.Value(1))
for x,z in [(15,65),(65,10),(65,65),(120,10),(120,65)]:
 post=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,-6,z),gp_Dir(0,1,0)),3,6).Shape()
 op=BRepAlgoAPI_Common(post,assembly);op.Build();props=GProp_GProps();BRepGProp.VolumeProperties_s(op.Shape(),props)
 rows.append({'cadCenter':[x,0,z],'booleanDone':op.IsDone(),'intersectionMM3':abs(props.Mass())});print(rows[-1],flush=True)
Path('work/torso-flat/projector-contact-assembly.json').write_text(json.dumps({'source':str(file),'scope':'Five radius3 underside posts against whole official REVD assembly. Touching base is intentional. Zero common volume does not establish retention or strength.','rows':rows},indent=2))
