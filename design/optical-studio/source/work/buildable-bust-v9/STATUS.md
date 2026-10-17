# v9 checkpoint — 2026-09-23

User selected printed parts + purchased hardware, JPY100000 cap, ST3215 retained, moving mass assumed2kg. Latest: measured135x75 projector base receiver, ignore modeled bottom screws. No installs/purchases/physical tests.

## Implemented candidate modules
- build-projector-tray.py:25 valid solids, receiver135.8x75.8, t5,4edge clips, no projector holes. CAD base thickness3.97 provisional. Tray contacts verified at4datum pads. Frame connection incomplete.
- build-motor-backplates.py: official rear pilot4holes, printed5mm plate. M2 thread spec not manufacturer confirmed. assemble-modules.py selects yaw-45/pitch+15/roll0 relativev8.58parts includingtray.
- build-mirror-mounts.py:38parts, printed backing/fixed plates,3adjustment fasteners and silicone. Frame absent.
- prepare-assembly.py:139part partial assembly. Old machined core retained. Do not release as completed.
- check-modules.mjs:99parts checked27poses, only intended projector contacts reported. Own module contacts excluded. No support frames; sampled chief rays, not finite pupil.
- New projector-mount-study.html: actual TI CAD +tray inspection, no old screw pattern. PROJECTOR_MOUNT.md records provisional dimensions and remaining gates. Main viewer links to this page but still displays v8 overall comparison.

## Rejected or unadopted
- Original9bulky motor clamp candidates all rejected.
- build-core-candidates.py:23valid thickenedcore solids but collide with projector, bridges, rollbelt/oldretainers. Do not adopt.
- R5printedstrut routing failed; searches interrupted. No finalized supportframe.
- metal-link-probe.json contains2surface-clear R3.3routes, not a rigid triangulated frame or selected stock hardware.

## Remaining
Full support frames/head attachments; printed core with purchased bearing/shafts/pulleyhubs; bodyfloor mounting; actual cable and cooling spaces; procurement/BOM/assembly instructions; load/thermal/retention tests. No physical strength certification.

Potential6814commercialbearing architecture only researched, not implemented. neck-aperture.json sampled chief rays show maxradius26.16at lowerneck; does not prove finite pupil or upperneck clearance. Do not represent this as adopted geometry.
