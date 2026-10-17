from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.StlAPI import StlAPI_Writer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
import json
r=STEPControl_Reader();r.ReadFile('work/buildable-bust-v2/st3215-cad/ST3215.step');r.TransferRoots();e=TopExp_Explorer(r.OneShape(),TopAbs_SOLID);rows=[];i=0
while e.More():
 s=TopoDS.Solid_s(e.Current());b=Bnd_Box();BRepBndLib.Add_s(s,b);bb=b.Get();i+=1
 if bb[0]<-25<bb[3] and bb[1]>4:
  rows.append({'solid':i,'bounds':bb});BRepMesh_IncrementalMesh(s,.04,False,.15,True).Perform();w=StlAPI_Writer();w.ASCIIMode=False;w.Write(s,f'work/buildable-bust-v6/solid-{i}.stl')
 e.Next()
print(json.dumps(rows,indent=2));open('work/buildable-bust-v6/horn-audit.json','w').write(json.dumps(rows,indent=2))
