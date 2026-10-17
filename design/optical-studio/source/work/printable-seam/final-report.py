from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
r=Path('outputs/danya-simulator/assets/projection-prototype/printable-seam');p=r/'actual-wall-slice-verification.json';d=json.loads(p.read_text());d['method']='Model extrusion centerlines within0.25mm of sampled wall transects in3D. Carrier:90positions x3stations=270, fixed generated inner/outer wall. Heads:90radial positions x2stations, split byZ. Actual final STL surface intersections define the wall; search backward by0.25mm up to8mm when the nominal line falls outside the trimmed rim. Two lower-head nominal lines have no closed wall intersection near the open neck and are reported separately, not counted as successful samples. Upper123/123, lower54valid wall transects. Finite samples, not continuous proof or physical print.';d['headProbeGeometry']=json.loads((r/'actual-head-transects.json').read_text());d['sourceSHA256']={n:hashlib.sha256((r/(n+'.stl')).read_bytes()).hexdigest()for n in ['head-upper','head-lower','magnet-carrier']};p.write_text(json.dumps(d,indent=2))
im=Image.new('RGB',(1100,340),'#f3f6f8');dr=ImageDraw.Draw(im)
for j,n in enumerate(['magnet-carrier','head-upper','head-lower']):
 row=d['results'][n];dr.text((20,20+j*100),n+' : '+str(row['hasModelExtrusionWithin025MM'])+'/'+str(row['sampleCount'])+' sampled walls have model extrusion',fill='#173640')
 for i,x in enumerate(row['distancesMM']):dr.rectangle((20+i*3.8,48+j*100,22+i*3.8,78+j*100),fill='#128a7d'if x<=.25 else'#df6040')
im.save(r/'verified-slice-paths.png')
