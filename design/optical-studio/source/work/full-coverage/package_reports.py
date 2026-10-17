import json,hashlib,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;r=p.parents[1]/'outputs'/'danya-simulator'
d=json.loads((p/'design.json').read_text());grid=json.loads((p/'patch-validation-0.25.json').read_text());outer=json.loads((p/'outer-validation.json').read_text());qa=json.loads((p/'patch-qa.json').read_text());qa['face']=qa.pop('patch');selfq=json.loads((p/'self-intersections.json').read_text());selfq['face']=selfq.pop('patch');thickness=json.loads((p/'thickness.json').read_text())
assert grid['counts']=={'lit':grid['total']} and grid['oldCounts']=={'lit':309265} and outer['lit']==outer['samples']
assert not any(grid[k] for k in ['mirrorHead','mirrorFace','bodyHead','bodyFace'])
assert all(selfq[k]['nonAdjacentIntersections']==0 and qa[k]['boundaryEdges']==0 and qa[k]['nonManifoldEdges']==0 and len(qa[k]['components'])==1 for k in ['head','face'])
report={'date':'2026-09-21','definition':'Front facial area coverage, not optical efficiency. No unlit targets excluded. 100% requires every test sample lit. Image extent ratio is an overscan metric, not a coverage percentage.','state':d['state'],'designChanges':{'rearXExpansion':d['k'],'rearYExpansion':d['sy'],'rearNeckShiftMM':d['neckRearShiftMM'],'innerJointReliefMM':d['innerJointReliefMM'],'faceInnerRadialScale':.996},'frontGrid':grid,'outerSurface':outer,'topology':qa,'selfIntersectionCheck':selfq,'faceThicknessEstimate':thickness,'hashes':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (r/'assets/full-coverage-v1').glob('*.bin')}}
(r/'full-coverage-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(r/'full-coverage-design.json').write_text(json.dumps(d,indent=2));(r/'recommended-layout.json').write_text(json.dumps(d['state'],indent=2))
# Preserve a runnable verification script against the shipped binary meshes.
s=(p/'verify-patch.mjs').read_text().replace("../../outputs/danya-simulator/node_modules/",'./node_modules/').replace("../../outputs/danya-simulator/src/",'./src/').replace("root=new URL('../../outputs/danya-simulator/',import.meta.url)","root=dir").replace("'design.json'","'full-coverage-design.json'").replace("(original?'assets/':'')+name+'.'+key+'.bin'","(original?'assets/':'assets/full-coverage-v1/')+(name==='patch'?'face':name)+'.'+key+'.bin'")
s=s.replace("fs.writeFileSync(new URL('patch-validation-'+step+'.json',dir),JSON.stringify(report,null,2));",'')
s += "\nif(report.counts.lit!==report.total||report.oldCounts.miss||report.originalMissing||report.newHidden||report.mirrorHead||report.mirrorFace||report.bodyHead||report.bodyFace)process.exitCode=1;\n"
(r/'validate-full-coverage.mjs').write_text(s)
print('Reports saved. Assertions passed.')
