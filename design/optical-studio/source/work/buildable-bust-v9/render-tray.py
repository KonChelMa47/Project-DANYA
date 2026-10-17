import bpy,json,numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
root=Path('/Users/daniilmalchenko/Documents/Codex/2026-09-21/bl');out=root/'outputs/danya-simulator/assets/buildable-bust-v9'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
d=json.load(open(root/'work/buildable-bust-v8/datums.json'))['projector']['matrix'];mat=Matrix([[d[i+4*j] for j in range(4)]for i in range(4)]);mat=mat@Matrix.Translation(Vector((-111.62,-13.97,-81.33583308323333)));inv=mat.inverted()
for p in json.load(open(out/'projector-tray-candidates.json'))['parts']:
 bpy.ops.wm.stl_import(filepath=str(out/(p['name']+'.stl')));o=bpy.context.object;o.matrix_world=inv;o.name=p['name'];o.color=(.08,.55,.46,1) if p['material'].startswith('printed') else (.15,.18,.2,1) if 'silicone' in p['material'] else (.7,.72,.75,1)
cam=bpy.data.cameras.new('InspectionCamera');o=bpy.data.objects.new('InspectionCamera',cam);bpy.context.collection.objects.link(o);o.location=(215,155,165);direction=Vector((67,0,37))-o.location;o.rotation_euler=direction.to_track_quat('-Z','Y').to_euler();cam.type='ORTHO';cam.ortho_scale=240;bpy.context.scene.camera=o
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_shadows=True;s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.background_type='WORLD';s.world.color=(.05,.07,.08);s.render.resolution_x=1200;s.render.resolution_y=800;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.filepath=str(out/'projector-tray-preview.png');bpy.ops.render.render(write_still=True)
