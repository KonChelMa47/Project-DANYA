exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'smooth-head-v4';ref=loadpath(out/'outer-envelope.stl');tree=BVHTree.FromPolygons([v.co for v in ref.data.vertices],[p.vertices[:]for p in ref.data.polygons]);report={}
for name in ['head-upper','head-lower']:
 o=loadpath(out/(name+'.stl'));moved=0
 for v in o.data.vertices:
  p=v.co.copy();h,n,idx,d=tree.find_nearest(p)
  if d>.8 or p.x>42:continue
  # Exact mechanical magnet seats remain untouched.
  if any((p-Vector(c)).length<4.6 for c in [(41,-18.054,255),(41,18.054,255),(19.351208,-30.09,65.70844),(19.533585,30.09,64.76681)]):continue
  if abs(p.z-165)<.0001:h.z=165
  if abs(p.z-50.65)<.0001:h.z=50.65
  v.co=h;moved+=1
 print(name,moved,export(o,name),flush=True)
