import bpy, numpy as np, json, struct
from pathlib import Path
p=Path(__file__).resolve().parent;s=json.loads((p/'design.json').read_text());data=np.load(p/'unified.npz')
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
m=bpy.data.meshes.new('DANYA united shell');m.from_pydata(data['v'].tolist(),[],data['f'].tolist());m.update();shell=bpy.data.objects.new('united',m);bpy.context.collection.objects.link(shell)
# Closed angular clearance volume from the virtual projector, terminated before the mirror.
v=s['virtual'];poly=s['cone'];verts=[]
for x in [s['coneStartX'],s['coneReferenceX']]:
 t=(x-v[0])/(s['coneReferenceX']-v[0]);verts += [(x,v[1]+(y-v[1])*t,v[2]+(z-v[2])*t) for y,z in poly]
n=len(poly);faces=[tuple(reversed(range(n))),tuple(range(n,n*2))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
cm=bpy.data.meshes.new('Clear optical cone');cm.from_pydata(verts,[],faces);cm.update();cutter=bpy.data.objects.new('Clear optical cone',cm);bpy.context.collection.objects.link(cutter)
objects=[]
for name,operation in [('face','INTERSECT'),('head','DIFFERENCE')]:
 o=bpy.data.objects.new('DANYA-'+name,m.copy());bpy.context.collection.objects.link(o);bpy.context.view_layer.objects.active=o;o.select_set(True)
 mod=o.modifiers.new('Reassign light-blocking seam','BOOLEAN');mod.operation=operation;mod.solver='EXACT';mod.object=cutter
 bpy.ops.object.modifier_apply(modifier=mod.name);o.select_set(False);objects.append(o);print(name,len(o.data.vertices),len(o.data.polygons),flush=True)
 o.data.calc_loop_triangles();v=np.empty(len(o.data.vertices)*3,np.float32);o.data.vertices.foreach_get('co',v);f=np.empty(len(o.data.loop_triangles)*3,np.int32);o.data.loop_triangles.foreach_get('vertices',f);v=v.reshape(-1,3);f=f.reshape(-1,3)
 np.savez(p/f'{name}-design.npz',v=v,f=f)
 for suffix,a in [('positions',v.astype('<f4')),('indices',f.astype('<u4'))]:(p/f'{name}.{suffix}.bin').write_bytes(a.tobytes())
 mat=bpy.data.materials.new(name);mat.diffuse_color=(.68,.75,.72,1) if name=='head' else (.85,.74,.64,1);o.data.materials.append(mat)
bpy.data.objects.remove(shell,do_unlink=True);bpy.data.objects.remove(cutter,do_unlink=True)
bpy.ops.wm.save_as_mainfile(filepath=str(p/'full-coverage-design.blend'))
