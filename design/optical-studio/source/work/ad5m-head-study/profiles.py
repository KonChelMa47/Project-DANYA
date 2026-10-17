import json
from pathlib import Path
root=Path('/Applications/Orca-Flashforge.app/Contents/Resources/profiles');out=Path('work/ad5m-head-study');index={}
for p in root.rglob('*.json'):
 try:d=json.loads(p.read_text());index[d.get('name',p.stem)]=p
 except:pass
def resolve(name):
 d=json.loads(index[name].read_text());base=resolve(d['inherits'])if d.get('inherits')else{};base.update(d);base.pop('inherits',None);return base
for typ,name in [('machine','Flashforge Adventurer 5M 0.4 Nozzle'),('process','0.20mm Standard @Flashforge AD5M 0.4 Nozzle'),('filament','Flashforge Generic PLA')]:
 d=resolve(name)
 if typ=='process':d.update({'enable_support':'1','support_type':'normal(auto)','support_on_build_plate_only':'0','support_threshold_angle':'30','support_base_pattern':'rectilinear','support_base_pattern_spacing':'2.5','support_top_z_distance':'0.2','support_object_xy_distance':'0.35','support_expansion':'0','brim_type':'no_brim','brim_width':'0','wall_loops':'4','sparse_infill_density':'15%','print_sequence':'by layer'})
 (out/(typ+'.json')).write_text(json.dumps(d,indent=2))
