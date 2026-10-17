import bpy, math, json, struct
from pathlib import Path
from mathutils import Vector
root=Path('/Users/daniilmalchenko/Documents/Codex/2026-09-21/bl');out=root/'outputs/danya-simulator/assets/buildable-bust-v2';out.mkdir(exist_ok=True)
spec=json.loads((root/'work/buildable-bust-v2/joints.json').read_text());bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def box(name,center,size,angle):
 bpy.ops.mesh.primitive_cube_add(size=1,location=center);o=bpy.context.object;o.name=name;o.dimensions=size;o.rotation_euler.z=angle;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);return o
def boolean(o,tool,operation):
 bpy.context.view_layer.objects.active=o;m=o.modifiers.new(operation,'BOOLEAN');m.operation=operation;m.solver='EXACT';m.object=tool;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(tool,do_unlink=True)
def cylinder(center,axis,r,depth):
 bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=r,depth=depth,location=center);o=bpy.context.object;o.rotation_mode='QUATERNION';o.rotation_quaternion=Vector((0,0,1)).rotation_difference(Vector(axis));return o
reports=[]
for part in spec['parts']:
 name=part['name'];bpy.ops.wm.stl_import(filepath=str(root/'outputs/danya-simulator/assets/buildable-bust-v1'/f'{name}.stl'));o=bpy.context.object;o.name=name
 for j in part['joints']:
  boolean(o,box('lug',j['center'],j['size'],j['angle']),'UNION')
 for j in part['joints']:
  boolean(o,cylinder(j['holeCenter'],j['axis'],1.7,30),'DIFFERENCE')
 # Horizontal parting tolerance: subtract 0.15mm each side of shared plane.
 if 'lower' in name:boolean(o,box('gap',[15,0,-95.075],[600,600,.15],0),'DIFFERENCE')
 else:boolean(o,box('gap',[15,0,-94.925],[600,600,.15],0),'DIFFERENCE')
 bpy.context.view_layer.objects.active=o;bpy.ops.object.select_all(action='DESELECT');o.select_set(True)
 bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.mesh.normals_make_consistent(inside=False);bpy.ops.object.mode_set(mode='OBJECT')
 mesh=o.data;edges={};adj=[set()for v in mesh.vertices]
 for p in mesh.polygons:
  for a,b in p.edge_keys:edges[(a,b)]=edges.get((a,b),0)+1;adj[a].add(b);adj[b].add(a)
 seen=set();components=0
 for i in range(len(adj)):
  if i in seen:continue
  components+=1;stack=[i];seen.add(i)
  while stack:
   for k in adj[stack.pop()]:
    if k not in seen:seen.add(k);stack.append(k)
 bad=sum(v!=2 for v in edges.values());bbox=[list(v)for v in o.bound_box];mins=[min(v[i]for v in bbox)for i in range(3)];maxs=[max(v[i]for v in bbox)for i in range(3)];size=[maxs[i]-mins[i]for i in range(3)]
 bpy.ops.wm.stl_export(filepath=str(out/f'{name}-assembly-coordinates.stl'),export_selected_objects=True)
 o.location-=Vector(mins)
 bpy.ops.wm.stl_export(filepath=str(out/f'{name}-print.stl'),export_selected_objects=True)
 reports.append(dict(name=name,sizeMM=size,connectedComponents=components,boundaryOrNonmanifoldEdges=bad,fitsAD5MWith5MMBrim=size[0]+10<=220 and size[1]+10<=220 and size[2]<=220,jointCount=len(part['joints'])))
 bpy.data.objects.remove(o,do_unlink=True)
(out/'shell-audit.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
