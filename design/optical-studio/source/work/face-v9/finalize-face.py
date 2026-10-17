import bpy,bmesh,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/'face-tips-restored.stl'));o=bpy.context.object;rep=export(o,'face-final');(out/'final-face-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True);assert rep['components']==1 and rep['nonManifoldEdges']==0
