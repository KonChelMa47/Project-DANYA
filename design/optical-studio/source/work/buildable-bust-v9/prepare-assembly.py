from pathlib import Path
import json,shutil
old=Path('outputs/danya-simulator/assets/buildable-bust-v8');out=old.with_name('buildable-bust-v9');parts=[]
for p in json.load(open(old/'parts.json'))['parts']:
 n=p['name']
 if p['parent']=='fixture' or n.startswith(('frame-','head-mount-','motor-','fastener-','projector-','mirror-','foldMirror-','floor-spacer-','roll-idler-carrier-')):continue
 parts.append(p)
 for ext in ['.stl','.step']:shutil.copy2(old/(n+ext),out/(n+ext))
for filename in ['mount-modules.json','mirror-mounts.json']:parts.extend(json.load(open(out/filename))['parts'])
for filename in ['transmission.json','pitch-belt.stl','roll-belt.stl','gear-layout.json']:
 if (old/filename).exists():shutil.copy2(old/filename,out/filename)
(out/'parts.json').write_text(json.dumps({'status':'v9 engineering work in progress; printed support frame absent; retained legacy core not yet converted','parts':parts},indent=2))
print('Candidate assembly',len(parts),'components')
