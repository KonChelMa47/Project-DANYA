exec(open('work/face-v10/build.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
import sys
out=r/'face-v11';reports={}
for name in sys.argv[sys.argv.index('--')+1:]:
 o=load(out/(name+'.stl'));bm=bmesh.new();bm.from_mesh(o.data);left=set(bm.verts);comps=[]
 while left:
  comp={left.pop()};stack=list(comp)
  while stack:
   v=stack.pop()
   for e in v.link_edges:
    w=e.other_vert(v)
    if w in left:left.remove(w);comp.add(w);stack.append(w)
  comps.append(comp)
 comps.sort(key=len,reverse=True);sizes=[]
 for c in comps[1:]:
  dims=[max(v.co[k]for v in c)-min(v.co[k]for v in c)for k in range(3)];assert max(dims)<1 or (name=='head-lower-release' and max(dims)<10 and min(dims)<1 and max(v.co.z for v in c)<57), dims;sizes.append(dims)
 bmesh.ops.delete(bm,geom=[v for c in comps[1:]for v in c],context='VERTS');bm.to_mesh(o.data);bm.free();rep=export(o,name);rep['removedFragmentDimensionsMM']=sizes;reports[name]=rep;print(name,rep,flush=True);bpy.data.objects.remove(o,do_unlink=True)
(out/('clean-'+sys.argv[-1]+'.json')).write_text(json.dumps(reports,indent=2))
