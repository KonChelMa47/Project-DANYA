import bpy,bmesh,json,math
from pathlib import Path
from mathutils import Matrix,Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'prototype-v7';meta=json.load(open(v/'parts.json'));(v/'print').mkdir(exist_ok=True);rows=[]
for p in meta['parts']:
 name=p['name']
 if 'glass' in name or '-nut-'in name or '-bond-pad-'in name:continue
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(v/(name+'.stl')));o=bpy.context.object
 if name.startswith('head-column'):M=Matrix.Rotation(math.pi/2,4,'X')
 elif name=='mirror-cutting-master':M=Matrix.Identity(4)
 else:
  key='foldMirror'if'foldMirror'in name else'mirror';m=meta['matrices'][key];M=Matrix([[m[j*4+i]for j in range(4)]for i in range(4)]).inverted()
 if name.endswith('-backing'):M=Matrix.Rotation(math.pi,4,'X')@M
 for vv in o.data.vertices:vv.co=M@vv.co
 lo=Vector([min(vv.co[k]for vv in o.data.vertices)for k in range(3)]);hi=Vector([max(vv.co[k]for vv in o.data.vertices)for k in range(3)]);S=Matrix.Translation(Vector((-(hi.x+lo.x)/2,-(hi.y+lo.y)/2,-lo.z)));M=S@M
 for vv in o.data.vertices:vv.co=S@vv.co
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.normal_update();bad=sum(not e.is_manifold for e in bm.edges);unseen=set(bm.verts);components=0
 while unseen:
  components+=1;stack=[unseen.pop()]
  while stack:
   vv=stack.pop()
   for e in vv.link_edges:
    nn=e.other_vert(vv)
    if nn in unseen:unseen.remove(nn);stack.append(nn)
 over=[f for f in bm.faces if f.normal.z<-.708 and min(vv.co.z for vv in f.verts)>.001];area=sum(f.calc_area()for f in over);bm.to_mesh(o.data);bm.free();assert bad==0 and components==1,(name,bad,components)
 if name.startswith('head-column'):assert area<.001,(name,area)
 bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(v/'print'/(name+'.stl')),export_selected_objects=True)
 rows.append({'name':name,'dimensionsMM':list(hi-lo),'components':components,'nonManifoldEdges':bad,'unsupportedDownwardAreaOver45degMM2':area,'assemblyToPrintMatrix':[M[i][j]for j in range(4)for i in range(4)],'physicalPrintTest':False})
(v/'print-check.json').write_text(json.dumps(rows,indent=2));print(rows)
