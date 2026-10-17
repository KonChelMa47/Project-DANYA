exec(open(__file__.replace('export_cad.py','read_step.py')).read())
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
import array
out=p/'cad-mesh';out.mkdir(exist_ok=True);manifest={};lens=[]
wanted={'3010-EVM-G2-SS-TOP-ASSEMBLY':'kit','900-0021-00 H6':'engine','3010-EVM-G2-ELECT-ASM':'electronics','DLPDLCR3010EVM-G2_MainBoard':'mainboard','DLPDLCR3010EVM-G2_DisplayBoard':'driverboard','P5_32FIN_ASM':'heatsink','259-1557-ND':'fan'}
def export(shape,key):
 print('Meshing',key,flush=True);BRepMesh_IncrementalMesh(shape,.12,False,.25,True);verts=array.array('f');faces=array.array('I');ex=TopExp_Explorer(shape,TopAbs_FACE)
 while ex.More():
  face=TopoDS.Face_s(ex.Current());location=TopLoc_Location();mesh=BRep_Tool.Triangulation_s(face,location)
  if mesh:
   off=len(verts)//3;t=location.Transformation()
   for i in range(1,mesh.NbNodes()+1):
    q=mesh.Node(i).Transformed(t);verts.extend([q.X(),q.Y(),q.Z()])
   for i in range(1,mesh.NbTriangles()+1):
    ids=list(mesh.Triangle(i).Get())
    if face.Orientation()==TopAbs_REVERSED:ids[1],ids[2]=ids[2],ids[1]
    faces.extend([off+j-1 for j in ids])
  if key=='engine':
   surf=BRepAdaptor_Surface(face)
   if surf.GetType()==GeomAbs_Cylinder:
    c=surf.Cylinder();a=c.Axis();q=a.Location();d=a.Direction();lens.append({'radius':c.Radius(),'point':[q.X(),q.Y(),q.Z()],'axis':[d.X(),d.Y(),d.Z()],'u':[surf.FirstUParameter(),surf.LastUParameter()],'v':[surf.FirstVParameter(),surf.LastVParameter()]})
  ex.Next()
 (out/(key+'.positions.bin')).write_bytes(verts.tobytes());(out/(key+'.indices.bin')).write_bytes(faces.tobytes());bounds=[[min(verts[k::3]) for k in range(3)],[max(verts[k::3]) for k in range(3)]];manifest[key]={'vertices':len(verts)//3,'triangles':len(faces)//3,'bounds':bounds};print(key,manifest[key],flush=True)
def collect(label,loc):
 ref=TDF_Label()
 if tool.IsReference_s(label):tool.GetReferredShape_s(label,ref);loc=loc.Multiplied(tool.GetLocation_s(label));target=ref
 else:target=label
 n=name(target)
 if n in wanted:
  key=wanted[n]
  if key in manifest:key+=str(len([k for k in manifest if k.startswith(key)]))
  export(tool.GetShape_s(target).Moved(loc),key)
 children=TDF_LabelSequence()
 if tool.GetComponents_s(target,children):
  for i in range(1,children.Length()+1):collect(children.Value(i),loc)
for i in range(1,roots.Length()+1):collect(roots.Value(i),TopLoc_Location())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'engine-cylinders.json').write_text(json.dumps(lens,indent=2))
