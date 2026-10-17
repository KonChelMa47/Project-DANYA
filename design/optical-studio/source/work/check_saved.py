import bpy,numpy as np,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
bpy.ops.wm.open_mainfile(filepath=str(root/'outputs/robot_shell_repaired.blend'))
result={}
for name in ['DANYA-face-back','DANYA-face-shell']:
 m=bpy.data.objects[name].data;expected=np.load(root/'work'/f'{name}-repaired.npz')
 v=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get('co',v);v=v.reshape(-1,3)
 m.calc_loop_triangles();f=np.empty(len(m.loop_triangles)*3,dtype=np.int32);m.loop_triangles.foreach_get('vertices',f);f=f.reshape(-1,3)
 assert np.array_equal(v,expected['v']) and np.array_equal(f,expected['f']),name
 result[name]={'saved_geometry_matches_verified_geometry':True,'vertices':len(v),'triangles':len(f),'custom_normals':m.has_custom_normals}
path=root/'outputs/robot_shell_validation.json';r=json.loads(path.read_text());r['saved_file_check']=result;path.write_text(json.dumps(r,indent=2))
print(json.dumps(result),flush=True)
