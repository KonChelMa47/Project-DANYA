exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
v=src/'split-head-v6';info=json.load(open(v/'head-split-check.json'));trees={}
for n in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(v/(n+'.stl')));o=bpy.context.object;trees[n]=BVHTree.FromPolygons([vv.co for vv in o.data.vertices],[p.vertices[:]for p in o.data.polygons])
rows=[]
for p in info['flanges']:
 x,y,z=p['boltCenter'];ws=[];ns=[]
 for angle in np.arange(0,math.tau,math.tau/24):
  for rr in [2.5,3.5,4.3]:
   h=trees['head-upper'].ray_cast(Vector((x+rr*math.cos(angle),y+rr*math.sin(angle),175.01)),Vector((0,0,-1)),30)[0];assert h is not None;ws.append(abs(h.z-175))
  h=trees['head-lower'].ray_cast(Vector((x+2.8*math.cos(angle),y+2.8*math.sin(angle),160.49)),Vector((0,0,1)),30)[0];assert h is not None;ns.append(abs(h.z-160.5))
 rows.append({'angle':p['angle'],'washerSeatErrorMM':max(ws),'nutSeatErrorMM':max(ns)})
report={'seats':rows,'splitPlaneMM':165,'lowerTopMM':info['head-lower']['worldBounds'][2][1],'upperBottomMM':info['head-upper']['worldBounds'][2][0],'nominalSeamGapMM':info['head-upper']['worldBounds'][2][0]-info['head-lower']['worldBounds'][2][1],'loadPath':'M4 head -> washer -> upper flange10mm -> lower flange -> hex nut seated under4.5mm shoulder. Side gussets and2mm wall liners distribute into shell. Pins align only.','physicalTests':False};(v/'joint-check.json').write_text(json.dumps(report,indent=2));print(report)
assert all(max(p['washerSeatErrorMM'],p['nutSeatErrorMM'])<.002 for p in rows)and abs(report['nominalSeamGapMM'])<.001
