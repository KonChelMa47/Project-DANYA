exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
for name in ['head-upper','head-lower']:
 o=load(out/(name+'-covered.stl'))
 if name=='head-upper':cut(o,box((81.55,0,290),(80,43,80)))
 else:cut(o,box((63.75,0,16.65),(80,72,80)))
 print(export(o,name+'-covered'),flush=True)
