from pathlib import Path
import json
r=Path('outputs/danya-simulator/assets/projection-prototype/unified-head-v5')
p=r/'print-transforms.json';d=json.loads(p.read_text());d['face']={'R':[[0,1,0],[0,0,1],[1,0,0]],'note':'Screen exterior faces upward; pose only, not sliced or support settings.'};p.write_text(json.dumps(d,indent=2))
p=Path('work/unified-head/sweep.mjs');s=p.read_text().replace('stepMM:1,','stepMM:1,nearStartStepMM:.25,nearStartTravelMM:15,');p.write_text(s)
s=Path('work/smooth-head/beauty.py').read_text().replace('work/smooth-head','work/unified-head').replace("(r/'assembly'if n=='face'else r/'solid-seam-v2'if n=='magnet-carrier'else r/'smooth-head-v4')","(r/'unified-head-v5')");Path('work/unified-head/beauty.py').write_text(s)
