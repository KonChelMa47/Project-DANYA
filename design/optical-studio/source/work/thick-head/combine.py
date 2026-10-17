exec(open('work/thick-head/thicken.py').read().split("head=loadpath(src/")[0])
head=loadpath(out/'jointless.stl');band=loadpath(out/'outer-band-fixed.stl');op(band,loadpath(src/'printable-seam/thin-lip-removal-tool.stl'),'DIFFERENCE')
def box(c,d):
 bpy.ops.mesh.primitive_cube_add(size=1,location=c);o=bpy.context.object;o.dimensions=d;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
op(band,box((0,0,-74.675),(600,600,250.65)),'DIFFERENCE');n=Vector((-.45,0,1)).normalized();tool=box(Vector((0,0,48.4))-n*300,(1000,1000,600));tool.rotation_euler[1]=-math.atan(.45);op(band,tool,'DIFFERENCE');print('BAND CUT',export(band,'outer-band'),flush=True);op(head,band,'UNION');print('THICK HEAD',export(head,'thick-head'),flush=True)
