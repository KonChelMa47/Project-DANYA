# RELEASED unified-head-v5
Latest user requested all head parts and face-carrier interface refinement, not just rear head.
Final four root STLs: face, magnet-carrier, head-upper, head-lower. All closed, one connected component each, zero degenerate triangles, zero strict crossings/coplanar overlaps at documented tolerance.
Adopted meshes: face from face.py; head-upper-final.stl; head-lower-finish.stl; magnet-carrier-face-fit-clean.stl. Do not run experimental carrier.py / topology-only / global-clean alternatives as release generators. They contain rejected variants.
Carrier final construction: smooth band (9 axial x720 angular), only four local old magnet mounting regions retained, full MeshFix repair, rebuild lower seats with3.35mm pocket radius for5.9x2.9 magnets, forward supporting beams3x5mm at lower position, further repair, subtract face expanded .12mm normal for adhesive fit, repair. Actual final geometry-validation.json records SHA256.
Face only outer10mm perimeter changed(max2.914mm); all161079 inner triangles exactly preserved. No nose/eyes/mouth changes.
Rear outer shell structured fit, majorwalls~4.3mm, Z165 split and original hardware retained. Lower head magnet seats/ribs rebuilt. Uppermagnets retain partly open concealed seat onone side (28/32 radial rays; intended clipped outer outline); lower seats all32/32 clear.
Final checks: 255542 cached nominal paths per blocker, 0 blocked for carrier and both heads. Removal +X0..100mm, .25mm through15mm, then1mm:0collisions. New neutral assembly contacts0, old intended contacts retained. M4 bolts/nuts/pins clear, nutentry0blocked. Face-carrier surface intersections0 after .12mm normal relief.
No slicer or physical test; no installations/subagents.
Released page http://127.0.0.1:8769/assets/projection-prototype/unified-head-v5/index.html
ZIP DANYA-head-complete-v5.zip includes4STLs+4pose3MF and guide/checks.
FullZIP complete-latest/DANYA-prototype-ALL-PARTS-unified-head-v5.zip:36printparts+4templates,32otherprintSTLs unchanged.40modelhashes verified. Removed obsolete HEAD_NEW_SPLIT/PRINTABLE_SEAM/THICK_HEAD_V3/SMOOTH_HEAD_V4 guide files from newZIP.
Main simulator routesall4to v5. Face-preview data newlycomputedat v5/face-preview; scriptbaseupdated. Preview modulecacheversionv5. Otherstand/projector/mirrors fixed.
Blender renders checked in work/unified-head +page. Finalassembled.png removed small black triangular penetration artifact near cheek after face-fit. External seams remain deliberatepartboundaries.
Server existing customserver8769 retained, not plainhttp. open_in_codex newpage queued. CUA inventory earlier timedout; renderedSTLs visually verified with view_image. No live browser screenshot yet this turn.
