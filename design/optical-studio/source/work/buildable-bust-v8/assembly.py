import json,math
from pathlib import Path
from OCP.STEPControl import STEPControl_Reader,STEPControl_Writer,STEPControl_AsIs
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Trsf,gp_Ax1,gp_Pnt,gp_Dir,gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
root=Path('outputs/danya-simulator/assets/buildable-bust-v8');parts=json.loads((root/'parts.json').read_text())['parts']
b=BRep_Builder();c=TopoDS_Compound();b.MakeCompound(c)
for p in parts:
 if p['parent']=='fixture':continue
 r=STEPControl_Reader();assert r.ReadFile(str(root/(p['name']+'.step'))).value==1;r.TransferRoots();s=r.OneShape();t=gp_Trsf()
 if p['parent']=='pinion':t.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)),5.916666164260777)
 t.SetTranslationPart(gp_Vec(*p['xy'],0));b.Add(c,BRepBuilderAPI_Transform(s,t,True).Shape())
w=STEPControl_Writer();w.Transfer(c,STEPControl_AsIs);w.Write(str(root/'neck-unit-assembly.step'))
