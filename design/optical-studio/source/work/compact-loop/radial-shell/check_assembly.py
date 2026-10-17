import bpy,bmesh,numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def load(name,vpath,fpath):
 v=np.fromfile(vpath,'<f4').reshape(-1,3);f=np.fromfile(fpath,'<u4').reshape(-1,3);m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);return o
h=load('head',p/'bevel-0.6.positions.bin',p/'bevel-0.6.indices.bin');f=load('face',root/'work/adult-kit/original-face.positions.bin',root/'outputs/danya-simulator/assets/face.indices.bin')
bpy.context.view_layer.objects.active=h;mod=h.modifiers.new('Material intersection audit','BOOLEAN');mod.operation='INTERSECT';mod.solver='EXACT';mod.object=f;print('intersection begin',flush=True);bpy.ops.object.modifier_apply(modifier=mod.name);bm=bmesh.new();bm.from_mesh(h.data);r={'intersectionVertices':len(bm.verts),'intersectionFaces':len(bm.faces),'signedIntersectionVolumeMM3':bm.calc_volume(signed=True)};bm.free();(p/'assembly-overlap.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
