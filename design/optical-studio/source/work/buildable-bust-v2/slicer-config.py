import json
from pathlib import Path
base=Path('/Applications/Orca-Flashforge.app/Contents/Resources/profiles');paths={p.stem:p for p in base.rglob('*.json')};out=Path('work/buildable-bust-v2/slicer');out.mkdir(exist_ok=True)
def resolve(name):
 p=paths[name];d=json.loads(p.read_text());parent=d.get('inherits');res=resolve(parent) if parent else {};res.update(d);res.pop('inherits',None);return res
names=['Flashforge Adventurer 5M 0.4 Nozzle','0.20mm Standard @Flashforge AD5M 0.4 Nozzle','Flashforge PETG'];
for name,filename in zip(names,['machine.json','process.json','filament.json']):
 d=resolve(name)
 if filename=='process.json':d.update(wall_loops='4',sparse_infill_density='20%',enable_support='1',support_type='normal(auto)',support_on_build_plate_only='0',brim_width='5',brim_type='outer_only',outer_wall_speed='45',inner_wall_speed='80',top_surface_speed='40')
 (out/filename).write_text(json.dumps(d,indent=2))
