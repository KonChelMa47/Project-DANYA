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
p=Path(__file__).resolve().parent
file=next((p/'ti-cad').rglob('*TOP-ASSEMBLY_REVD.STEP'))
doc=TDocStd_Document(TCollection_ExtendedString('DANYA'));reader=STEPCAFControl_Reader();reader.SetNameMode(True);print('Read',reader.ReadFile(str(file)),flush=True);print('Transfer',reader.Transfer(doc),flush=True);tool=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main());roots=TDF_LabelSequence();tool.GetFreeShapes(roots);rows=[]
def name(label):
 a=TDataStd_Name()
 return a.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(),a) else ''
def entry(label):
 a=TCollection_AsciiString();TDF_Tool.Entry_s(label,a);return a.ToCString()
def walk(label,loc,depth,parent):
 ref=TDF_Label();isref=tool.IsReference_s(label)
 if isref:tool.GetReferredShape_s(label,ref);loc=loc.Multiplied(tool.GetLocation_s(label));target=ref
 else:target=label
 shape=tool.GetShape_s(target);shape=shape.Moved(loc);box=Bnd_Box();BRepBndLib.Add_s(shape,box);eid=entry(label);r={'id':eid,'parent':parent,'depth':depth,'name':name(target),'instance':name(label),'bounds':list(box.Get()),'assembly':tool.IsAssembly_s(target)};rows.append(r)
 if depth<3:print(' '*depth,eid,r['name'],r['bounds'],flush=True)
 children=TDF_LabelSequence()
 if tool.GetComponents_s(target,children):
  for i in range(1,children.Length()+1):walk(children.Value(i),loc,depth+1,eid)
for i in range(1,roots.Length()+1):walk(roots.Value(i),TopLoc_Location(),0,None)
(p/'cad-assembly-tree.json').write_text(json.dumps(rows,indent=2));print('Rows',len(rows),flush=True)
