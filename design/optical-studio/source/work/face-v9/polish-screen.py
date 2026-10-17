import bpy,bmesh,numpy as np,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
a=np.load(r/'face-v8/screen-data.npz');w=a['after'].copy();start=w.copy();f=a['faces'];innerf=a['innerFaces'];n=a['normals'];weights=np.zeros(len(w))
for center in [[13.5,-63.5,115],[38.4,-44,233.4]]:weights=np.maximum(weights,np.maximum(0,1-np.linalg.norm(w-center,axis=1)/4)**2)
mask=np.zeros(len(w),bool);mask[np.unique(innerf)]=True;weights[~mask]=0
edges=np.unique(np.sort(np.concatenate([innerf[:,[0,1]],innerf[:,[1,2]],innerf[:,[2,0]]]),axis=1),axis=0);i=np.r_[edges[:,0],edges[:,1]];j=np.r_[edges[:,1],edges[:,0]];count=np.bincount(i,minlength=len(w));amount=np.zeros(len(w))
for _ in range(60):
 avg=np.stack([np.bincount(i,weights=w[j,k],minlength=len(w))for k in range(3)],axis=1)/np.maximum(count,1)[:,None]
 step=np.maximum(0,np.sum((avg-w)*n,axis=1))*.65*weights
 step=np.minimum(step,.18-amount);amount+=step;w=start+amount[:,None]*n
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);m=bpy.data.meshes.new('screen');m.from_pydata(w.tolist(),[],f.tolist());m.update();o=bpy.data.objects.new('screen',m);bpy.context.collection.objects.link(o);bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(m);bm.free();o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(out/'screen-after.stl'),export_selected_objects=True)
(out/'screen-polish.json').write_text(json.dumps({'method':'Only inward material addition in two interior patches; exterior and original v8 thickness are not reduced. Smooth projected-normal Laplacian, maximum0.18mm local addition.','maximumAdditionalInwardMM':float(amount.max()),'affectedVertices':int((amount>1e-6).sum()),'centers':[[13.5,-63.5,115],[38.4,-44,233.4]],'radiusMM':4},indent=2));np.savez(out/'screen-data.npz',after=w,faces=f,outerFaces=a['outerFaces'])
