exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
out=src/'unified-head-v5'
def direct(a,b,mode):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
for name in ['head-upper','head-lower']:
 head=loadpath(out/(name+'.stl'))
 for part in (['mirror-glass']if name=='head-upper'else ['head-connect-0-nut','head-connect-3-bolt','head-connect-3-nut']):
  tool=loadpath(src/'assembly'/(part+'.stl'));vs=np.array([v.co[:]for v in tool.data.vertices]);c=vs.mean(0);_,axes=np.linalg.eigh((vs-c).T@(vs-c));local=(vs-c)@axes;ext=np.ptp(local,axis=0);local*=1+(.5 if part=='mirror-glass'else .25)/np.maximum(ext,.01)
  for v,p in zip(tool.data.vertices,c+local@axes.T):v.co=p
  direct(head,tool,'DIFFERENCE')
 print(name,export(head,name),flush=True)
