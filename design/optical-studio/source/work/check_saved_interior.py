import bpy,numpy as np,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
bpy.ops.wm.open_mainfile(filepath=str(root/'outputs/robot_shell_clean_interior.blend'))
for name,file in [('DANYA-face-back','interior_back.npz'),('DANYA-face-shell','DANYA-face-shell-repaired.npz')]:
 m=bpy.data.objects[name].data;v=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get('co',v);m.calc_loop_triangles();f=np.empty(len(m.loop_triangles)*3,dtype=np.int32);m.loop_triangles.foreach_get('vertices',f);expected=np.load(root/'work'/file)
 assert np.array_equal(v.reshape(-1,3),expected['v']) and np.array_equal(f.reshape(-1,3),expected['f'])
 assert not bpy.data.objects[name].hide_get()
 print(name+': saved geometry matches verified geometry; visible in saved file',flush=True)
path=root/'outputs/robot_shell_interior_validation.json';r=json.loads(path.read_text());r['saved_geometry_matches_verified_geometry']=True;path.write_text(json.dumps(r,indent=2))
