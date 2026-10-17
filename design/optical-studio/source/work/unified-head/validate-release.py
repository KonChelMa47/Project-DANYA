import json
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype/unified-head-v5')
def get(n):return json.loads((r/n).read_text())
for n,j in get('geometry-validation.json').items():
 assert j['nonManifoldEdges']==0,(n,'nonmanifold',j)
 assert j['degenerateTriangles']==0,(n,'degenerate',j)
 j=get(n+'-surface.json');assert j['crossingsOver01MM']==0,(n,j)
 assert j['coplanarOverlapsOver00001MM2']==0,(n,'overlap',j)
for n in ['head-upper','head-lower','magnet-carrier']:assert get(n+'-rays.json')['blockedSamples']==0,n
assert not get('face-removal-sweep.json')['collisions']
assert not get('hardware-fit.json')['surfaceContacts']
assert not any(j['blockedSteps']for j in get('nut-entry-check.json'))
assert not get('assembly-contact-check.json')['newContacts']
f=get('face-perimeter-change.json');assert f['innerTrianglesExactlyPreserved']==f['innerTrianglesChecked']
print('Release gates passed. Inspect components, magnet clearance and rendered appearance separately.')
