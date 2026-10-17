import bpy,bmesh,numpy as np,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v10';out.mkdir(exist_ok=True)
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v10'
def load(p):
 bpy.ops.wm.stl_import(filepath=str(p));return bpy.context.object
def union(a,b):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new('join','BOOLEAN');m.operation='UNION';m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
d=np.load(r/'face-v8/screen-data.npz');v=d['after'];f=d['faces'];cap=np.load('work/shape-preserved/face-original.npz')['cap'];cf=f[cap];ids=np.unique(cf);lookup=np.full(len(v),-1,dtype=int);lookup[ids]=np.arange(len(ids))+len(v)
w=v[ids].copy();w[:,0]-=6
edges={}
for tri in cf:
 for a,b in zip(tri,np.roll(tri,-1)):
  key=tuple(sorted((int(a),int(b))));edges.setdefault(key,[]).append((int(a),int(b)))
bridge=[]
for es in edges.values():
 if len(es)==1:
  a,b=es[0];bridge += [(a,b,int(lookup[b])),(a,int(lookup[b]),int(lookup[a]))]
ff=np.vstack([f[~cap],lookup[cf],np.array(bridge)]);vv=np.vstack([v,w]);m=bpy.data.meshes.new('continuous face rim');m.from_pydata(vv.tolist(),[],ff.tolist());m.update();o=bpy.data.objects.new('face-ring',m);bpy.context.collection.objects.link(o);clean(o);rep=export(o,'face-ring');print('ring',rep,flush=True)
reports={'ring':rep}
for i in range(4):
 arm=load(r/'face-v9'/f'face-holder-{i}.stl');union(o,arm);rep=export(o,f'face-joined-{i}');reports[f'join{i}']=rep;print('join',i,rep,flush=True)
(out/'build-report.json').write_text(json.dumps(reports,indent=2))
