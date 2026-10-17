import bpy,bmesh,numpy as np,json,struct,shutil
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion,Euler
p=Path(__file__).resolve().parent;root=p.parents[1]/'outputs'/'danya-simulator';out=root/'assets'/'adult-kit-v1';out.mkdir(parents=True,exist_ok=True);s=json.loads((p/'design.json').read_text())['state'];manifest={}
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
shells=[]
def rotation(values):
 a=[x*np.pi/180 for x in values];return Matrix.Rotation(a[0],3,'X')@Matrix.Rotation(a[1],3,'Y')@Matrix.Rotation(a[2],3,'Z')
def binary_stl(path,v,f):
 t=v[f];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20);a=np.zeros(len(f),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);a['n']=n;a['v']=t;path.write_bytes(b'Project DANYA full coverage design'.ljust(80,b'\0')+struct.pack('<I',len(f))+a.tobytes())
for name,src in [('head','head'),('face','face')]:
 v=np.fromfile(p/f'{src}.positions.bin','<f4').reshape(-1,3);f=np.fromfile(p/f'{src}.indices.bin','<u4').reshape(-1,3);m=bpy.data.meshes.new(name);m.from_pydata(v.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new('DANYA-'+name,m);bpy.context.collection.objects.link(o);shells.append(o)
 n=np.empty(v.size,np.float32);m.vertices.foreach_get('normal',n)
 for suffix,a in [('positions',v.astype('<f4')),('normals',n.astype('<f4')),('indices',f.astype('<u4'))]:(out/f'{name}.{suffix}.bin').write_bytes(a.tobytes())
 binary_stl(out/f'DANYA-{name}-full-coverage.stl',v,f);np.savez(p/f'{name}-final.npz',v=v,f=f)
 mat=bpy.data.materials.new(name);mat.diffuse_color=(.34,.42,.4,1) if name=='head' else (.82,.88,.84,1);m.materials.append(mat)
 for poly in m.polygons:poly.use_smooth=True
 manifest[name]={'vertices':len(v),'triangles':len(f),'bounds':[v.min(0).tolist(),v.max(0).tolist()]};o['role']='Opaque rear housing' if name=='head' else 'Original external facial surface at 110 percent; ray-aligned inner wall 0.297 to 1.1 mm normal thickness'
# Manufacturing reference for the custom front-surface mirror, optical plane at local Z=0.
mi=s['mirror'];mv=np.fromfile(p/'mirror.positions.bin','<f4').reshape(-1,3);mf=np.fromfile(p/'mirror.indices.bin','<u4').reshape(-1,3)
m=bpy.data.meshes.new('Convex spherical mirror');m.from_pydata(mv.tolist(),[],mf.tolist());m.update();o=bpy.data.objects.new('REFERENCE - convex mirror',m);bpy.context.collection.objects.link(o);o.location=mi['position'];o.rotation_mode='QUATERNION';o.rotation_quaternion=rotation(mi['rotation']).to_quaternion();o['curvature_radius_mm']=mi['curvatureRadius'];o['thickness_mm']=1.5
binary_stl(out/'DANYA-mirror-local.stl',mv,mf);mat=bpy.data.materials.new('Mirror');mat.diffuse_color=(.8,.68,.35,1);m.materials.append(mat)
for key in ['preMirror','foldMirror']:
 fold=s[key];bpy.ops.mesh.primitive_cube_add(size=1);o=bpy.context.object;o.name='REFERENCE - '+key;o.dimensions=(fold['width'],fold['height'],1.5);o.rotation_mode='QUATERNION';o.rotation_quaternion=rotation(fold['rotation']).to_quaternion();o.location=Vector(fold['position'])+rotation(fold['rotation'])@Vector((0,0,-.85));o.data.materials.append(mat)
pr=s['projector'];rot=rotation(pr['rotation']);ev=np.fromfile(root/'assets/tripo-projector/projector.positions.bin','<f4').reshape(-1,3);ef=np.fromfile(root/'assets/tripo-projector/projector.indices.bin','<u4').reshape(-1,3);em=bpy.data.meshes.new('TripoAI complete DLPDLCR3010EVM-G2');em.from_pydata(ev.tolist(),[],ef.tolist());em.update();o=bpy.data.objects.new('DLPDLCR3010EVM-G2 full module - base 135 x 75 mm',em);bpy.context.collection.objects.link(o);o.rotation_mode='QUATERNION';o.rotation_quaternion=rot.to_quaternion();o.location=pr['position'];o['note']='User TripoAI mesh calibrated to TI whole-module bounds; base measurement 135 x 75 mm agrees with official CAD. Includes boards and cooling. Local shape and exit pupil are approximate.'
mat=bpy.data.materials.new('Projector');mat.diffuse_color=(.18,.49,.56,1);o.data.materials.append(mat)
bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=.001;bpy.context.scene['layout_json']=json.dumps(s);bpy.context.scene['design_note']='Original exterior uniformly enlarged 110 percent; complete projector remains actual size. Two flat mirrors and custom R16.5 mm spherical mirror. Geometric simulation only; optical focus, tolerances, heat and cable routing remain unverified.'
for area in bpy.context.screen.areas:
 if area.type=='VIEW_3D':area.spaces.active.region_3d.view_distance=620;area.spaces.active.region_3d.view_location=Vector((-30,0,20))
bpy.ops.wm.save_as_mainfile(filepath=str(root/'DANYA-adult-full-projector.blend'));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
