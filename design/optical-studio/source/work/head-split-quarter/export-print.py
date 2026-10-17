exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'head-split-quarter';data=json.load(open(out/'head-split-check.json'));reports={};(out/'print').mkdir(exist_ok=True);(out/'assembly').mkdir(exist_ok=True)
for name in ['head-upper','head-lower']:
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/(name+'.stl')));o=bpy.context.object
 M=Matrix.Identity(4)if name=='head-upper'else Matrix.Rotation(math.pi,4,'X');q=[M@v.co for v in o.data.vertices];lo=Vector([min(v[k]for v in q)for k in range(3)]);hi=Vector([max(v[k]for v in q)for k in range(3)]);t=Vector((-(lo.x+hi.x)/2,-(lo.y+hi.y)/2,-lo.z));M=Matrix.Translation(t)@M;o.data.transform(M);bpy.ops.wm.stl_export(filepath=str(out/'print'/(name+'.stl')),export_selected_objects=True);reports[name]={'sizeMM':list(hi-lo),'assemblyToPrint':[list(row)for row in M],'orientation':'new split plane down'}
import shutil
for n in ['head-upper','head-lower']:shutil.copy2(out/(n+'.stl'),out/'assembly'/(n+'.stl'))
for n in ['head-alignment-pin-0','head-alignment-pin-1']:shutil.copy2(src/'print'/(n+'.stl'),out/'print'/(n+'.stl'))
(out/'print-layout.json').write_text(json.dumps(reports,indent=2));print(reports)
