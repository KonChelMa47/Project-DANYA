exec(open('work/head-connectors-v3/build-shell.py').read().split("# Smooth radial")[0])
bpy.data.objects.remove(head,do_unlink=True)
report=json.load(open(out/'shell-report.json'))
for name,key in [('head-neck-clean','neck'),('head-integrated','head')]:
 bpy.ops.wm.stl_import(filepath=str(out/(name+'.stl')));head=bpy.context.object
 exec(open('work/head-connectors-v3/build-shell.py').read().split('# Delete disconnected scraps')[1].split('\n',1)[1].split("report={'neck'")[0])
 report[key]=export(head,name);report['removedDisconnectedScrapVertices']=len(scraps);assert report[key]['components']==1 and report[key]['nonManifoldEdges']==0
 bpy.data.objects.remove(head,do_unlink=True)
(out/'shell-report.json').write_text(json.dumps(report,indent=2));print(report)
