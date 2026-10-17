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

from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
found=[]
def collect(label,loc):
 ref=TDF_Label()
 if tool.IsReference_s(label):tool.GetReferredShape_s(label,ref);loc=loc.Multiplied(tool.GetLocation_s(label));target=ref
 else:target=label
 if name(target)=='3010-EVM-G2-ENGINE-BASE-SS':
  shape=tool.GetShape_s(target).Moved(loc)
  from OCP.BRepClass3d import BRepClass3d_SolidClassifier
  from OCP.gp import gp_Pnt
  from OCP.TopAbs import TopAbs_IN,TopAbs_ON
  import math
  for x,z in [(15,65),(65,10),(65,65),(120,10),(120,65)]:
   from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
   from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut,BRepAlgoAPI_Common
   from OCP.BRepGProp import BRepGProp
   from OCP.GProp import GProp_GProps
   from OCP.gp import gp_Ax2,gp_Dir
   def volume(sh):
    props=GProp_GProps();BRepGProp.VolumeProperties_s(sh,props);return abs(props.Mass())
   disk=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,0,z),gp_Dir(0,1,0)),3,.05).Shape()
   missing=BRepAlgoAPI_Cut(disk,shape);missing.Build()
   post=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,-6,z),gp_Dir(0,1,0)),3,6).Shape()
   clash=BRepAlgoAPI_Common(post,shape);clash.Build()
   states=[]
   for r in [0,1,2,3]:
    for j in range(32):
     a=j*2*math.pi/32
     c=BRepClass3d_SolidClassifier(shape,gp_Pnt(x+r*math.cos(a),.05,z+r*math.sin(a)),1e-7)
     states.append(c.State() in (TopAbs_IN,TopAbs_ON))
   found.append({'cadCenter':[x,0,z],'radiusMM':3,'insideAtDepth005':sum(states),'samples':len(states),'passesSampledFootprint':all(states),'missingDiskVolumeMM3':volume(missing.Shape()),'postBaseIntersectionMM3':volume(clash.Shape()),'booleansDone':missing.IsDone() and clash.IsDone()})
  return
 children=TDF_LabelSequence()
 if tool.GetComponents_s(target,children):
  for i in range(1,children.Length()+1):collect(children.Value(i),loc)
for i in range(1,roots.Length()+1):collect(roots.Value(i),TopLoc_Location())
Path('work/torso-flat/projector-contact-footprint.json').write_text(json.dumps({'scope':'Radius3 contact disk: samples and CAD boolean of 0.05mm layer; 6mm underside post vs base. Whole assembly interference and retention remain unverified.','rows':found},indent=2));print(json.dumps(found,indent=2))
