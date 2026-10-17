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
 if entry(label) in ['0:1:1:1:18','0:1:1:1:19']:
  shape=tool.GetShape_s(target).Moved(loc);ex=TopExp_Explorer(shape,TopAbs_FACE);cylinders=[]
  while ex.More():
   face=TopoDS.Face_s(ex.Current());surf=BRepAdaptor_Surface(face)
   if surf.GetType()==GeomAbs_Cylinder:
    c=surf.Cylinder();axis=c.Axis();q=axis.Location();d=axis.Direction();cylinders.append({'radiusMM':c.Radius(),'origin':[q.X(),q.Y(),q.Z()],'axis':[d.X(),d.Y(),d.Z()],'u':[surf.FirstUParameter(),surf.LastUParameter()],'v':[surf.FirstVParameter(),surf.LastVParameter()],'reversedFace':face.Orientation()==TopAbs_REVERSED})
   ex.Next()
  t=loc.Transformation();found.append({'name':name(target),'transform':[[t.Value(i,j) for j in range(1,5)] for i in range(1,4)],'cylinders':cylinders});return
 children=TDF_LabelSequence()
 if tool.GetComponents_s(target,children):
  for i in range(1,children.Length()+1):collect(children.Value(i),loc)
for i in range(1,roots.Length()+1):collect(roots.Value(i),TopLoc_Location())
Path('work/torso-flat/projector-existing-screw-features.json').write_text(json.dumps({'source':str(file),'scope':'Base cylindrical surfaces in authoritative assembly coordinates. Face orientation and angular spans distinguish candidates; no thread or spare mounting availability assumed.','parts':found},indent=2));print(json.dumps(found,indent=2))
