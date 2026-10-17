exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=src/'printable-seam';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def op(a,b,mode):
 clean(a);clean(b);bpy.context.view_layer.objects.active=a;m=a.modifiers.new(mode,'BOOLEAN');m.operation=mode;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def box(c,d):
 bpy.ops.mesh.primitive_cube_add(size=1,location=c);o=bpy.context.object;o.dimensions=d;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
bpy.ops.wm.stl_import(filepath=str(out/'head-upper.stl'));head=bpy.context.object
bpy.ops.wm.stl_import(filepath=str(src/'head-split-quarter/head-upper.stl'));old=bpy.context.object;op(old,box((39.8,-18.054,256.5),(5,7,10)),'INTERSECT');op(head,old,'UNION');op(head,box((39,-18.054,258.6),(2.6,4,4.2)),'UNION');rep=export(head,'head-upper');(out/'seat-restoration.json').write_text(json.dumps(rep,indent=2));print(rep)
