import re,json
from pathlib import Path
p=Path('work/internal-projector/ti-cad/DLPDLCR3010EVM-G2_Mechanical/DLP002_DLPDLCR3010EVM-G2-ENGINE-BASE-SS.STEP')
s=p.read_text();entities={int(i):v.strip() for i,v in re.findall(r'#(\d+)\s*=\s*(.*?);',s,re.S)}
def refs(v): return [int(x) for x in re.findall(r'#(\d+)',v)]
def vector(k):return [float(x) for x in re.search(r'\(\s*\x27[^\x27]*\x27\s*,\s*\((.*?)\)',entities[k],re.S).group(1).split(',')]
rows=[]
for k,v in entities.items():
 if v.startswith('CYLINDRICAL_SURFACE'):
  axis=refs(v)[0];a=refs(entities[axis]);radius=float(v.rsplit(',',1)[1].split(')')[0]);rows.append({'entity':k,'radiusMM':radius,'origin':vector(a[0]),'axis':vector(a[1])})
report={'source':str(p),'scope':'Cylinder surfaces extracted from official base STEP; may be fillets, counterbores or holes. No fastener/thread or assembly transform inferred. Requires topology and assembly registration confirmation.','cylinders':rows}
Path('work/torso-flat/base-cylinder-features.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))
