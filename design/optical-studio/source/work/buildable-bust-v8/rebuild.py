from pathlib import Path
import json,shutil,subprocess,sys
root=Path('outputs/danya-simulator/assets/buildable-bust-v8');old=root.with_name('buildable-bust-v7')
for p in json.load(open(old/'parts.json'))['parts']:
 for ext in ['.step','.stl']:shutil.copy2(old/(p['name']+ext),root/(p['name']+ext))
for script in ['build-mounts.py','build-drive-supports.py','build-frame.py','build-fasteners.py','build-interfaces.py','assembly.py']:
 print('Building',script,flush=True)
 with open('work/buildable-bust-v8/'+script+'.log','w')as f:subprocess.run([sys.executable,'work/buildable-bust-v8/'+script],stdout=f,stderr=subprocess.STDOUT,check=True)
parts=json.load(open(root/'parts.json'))['parts'];assert len({p['name']for p in parts})==len(parts);assert all(p['validSolid']and p['solidCount']==1 for p in parts)
print('VALID',len(parts),'unique single-solid components',flush=True)
