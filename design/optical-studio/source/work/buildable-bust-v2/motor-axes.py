from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
import json
r=STEPControl_Reader();r.ReadFile('work/buildable-bust-v2/st3215-cad/ST3215.step');r.TransferRoots();e=TopExp_Explorer(r.OneShape(),TopAbs_FACE);rows=[]
while e.More():
 f=TopoDS.Face_s(e.Current());a=BRepAdaptor_Surface(f)
 if a.GetType()==GeomAbs_Cylinder:
  c=a.Cylinder();p=c.Location();d=c.Axis().Direction();row=[round(c.Radius(),3),[round(p.X(),3),round(p.Y(),3),round(p.Z(),3)],[round(d.X(),3),round(d.Y(),3),round(d.Z(),3)]]
  if row not in rows:rows.append(row)
 e.Next()
print(json.dumps(rows))
