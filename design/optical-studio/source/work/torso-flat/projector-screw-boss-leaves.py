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

leaves=[]
def collect(label,loc):
 ref=TDF_Label()
 if tool.IsReference_s(label):
  tool.GetReferredShape_s(label,ref);loc=loc.Multiplied(tool.GetLocation_s(label));target=ref
 else:target=label
 children=TDF_LabelSequence()
 if tool.GetComponents_s(target,children):
  for i in range(1,children.Length()+1):collect(children.Value(i),loc)
 else:
  shape=tool.GetShape_s(target).Moved(loc);bb=Bnd_Box();BRepBndLib.Add_s(shape,bb)
  if not bb.IsVoid():leaves.append((name(target),shape,bb.Get()))
for i in range(1,roots.Length()+1):collect(roots.Value(i),TopLoc_Location())

from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.gp import gp_Ax2,gp_Dir
for i,x in enumerate([13.742,61.742]):
 z=48.0002;origin=[x-4,-10,z-4];size=[8,10,8]
 axis=gp_Ax2(gp_Pnt(x,-10,z),gp_Dir(0,1,0))
 outer=BRepPrimAPI_MakeCylinder(axis,4,10).Shape()
 bore=BRepPrimAPI_MakeCylinder(axis,1.2,10).Shape()
 pocket=BRepPrimAPI_MakeCylinder(axis,2.2,2).Shape()
 box=BRepAlgoAPI_Cut(BRepAlgoAPI_Cut(outer,bore).Shape(),pocket).Shape();hits=[]
 for label,shape,b in leaves:
  if any(b[j]>origin[j]+size[j]+1e-5 or b[j+3]<origin[j]-1e-5 for j in range(3)):continue
  op=BRepAlgoAPI_Common(box,shape);op.Build();props=GProp_GProps();BRepGProp.VolumeProperties_s(op.Shape(),props)
  hits.append({'name':label,'done':op.IsDone(),'volumeMM3':abs(props.Mass())})
 rows.append({'centerCAD':[x,-5,z],'radiusMM':4,'heightMM':10,'boreRadiusMM':1.2,'counterboreRadiusMM':2.2,'counterboreDepthMM':2,'checkedParts':hits});print(rows[-1],flush=True)
 Path('work/torso-flat/projector-screw-boss-leaves.json').write_text(json.dumps({'scope':'Two underside bosses centered on occupied existing screw axes. Intended replacement with +10mm longer under-head length to preserve original tip Y8. Thread engagement and replacement hardware qualification pending; no CAD hardware modified.','leafCount':len(leaves),'complete':len(rows)==2,'rows':rows},indent=2))
