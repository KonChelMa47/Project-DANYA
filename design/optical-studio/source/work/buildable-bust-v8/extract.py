import json
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from pathlib import Path
for key,path in [('motor','work/buildable-bust-v2/st3215-cad/ST3215.step'),('base','work/internal-projector/ti-cad/DLPDLCR3010EVM-G2_Mechanical/DLP002_DLPDLCR3010EVM-G2-ENGINE-BASE-SS.STEP')]:
 r=STEPControl_Reader();r.ReadFile(path);r.TransferRoots();e=TopExp_Explorer(r.OneShape(),TopAbs_FACE);rows=[]
 while e.More():
  a=BRepAdaptor_Surface(TopoDS.Face_s(e.Current()))
  if a.GetType()==GeomAbs_Cylinder:
   c=a.Cylinder();p=c.Location();d=c.Axis().Direction();row=dict(r=round(c.Radius(),4),p=[round(p.X(),4),round(p.Y(),4),round(p.Z(),4)],d=[round(d.X(),4),round(d.Y(),4),round(d.Z(),4)],v=[round(a.FirstVParameter(),4),round(a.LastVParameter(),4)])
   if row not in rows:rows.append(row)
  e.Next()
 Path('work/buildable-bust-v8/'+key+'-cylinders.json').write_text(json.dumps(rows,indent=2));print(key,rows)
