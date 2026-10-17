exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=root/'outputs/danya-simulator/assets/head-connectors-v3';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(out/'head-integrated.stl'));head=bpy.context.object;rows=[]
for name in [f'head-connect-{i}-nut'for i in range(4)]+['static-head-arm-left','static-head-arm-right','motorized-head-arm-left','motorized-head-arm-right','printed-mirror-fixed-plate','mirror-head-bolt-0','mirror-head-bolt-1']:
 p=out/(name+'.stl');p=p if p.exists()else src/'assembly'/(name+'.stl');bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;clean(o)
 if 'head-arm-' in name:
  sg=-1 if name.endswith('left')else 1
  for vv in o.data.vertices:vv.co+=Vector((.0045,-sg*.00893028555,0))
 if name=='printed-mirror-fixed-plate':
  mm=json.load(open(root/'work/buildable-bust-v8/datums.json'))['mirror']['matrix'];delta=Vector(mm[8:11])*.01
  for vv in o.data.vertices:vv.co+=delta
 boolean(o,head,'INTERSECT',remove=False);bm=bmesh.new();bm.from_mesh(o.data);volume=abs(bm.calc_volume(signed=True));bm.free();rows.append({'part':name,'headOverlapMM3':volume,'numericalSeparationMM':.01 if 'head-arm-'in name or name=='printed-mirror-fixed-plate'else 0});print(rows[-1],flush=True);bpy.data.objects.remove(o,do_unlink=True)
(out/'mating-check.json').write_text(json.dumps(rows,indent=2))
