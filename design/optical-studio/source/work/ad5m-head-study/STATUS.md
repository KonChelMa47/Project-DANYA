# AD5M one-plate rear head — delivered 2026-09-25
User clarified: TWO rear-head parts together on ONE AD5M plate (not one-piece head).
Solution: preserve all v12 assembly geometry; orient upper +90°Y, lower +90°X then -10°Z; arrange concave footprints on bed. Nominal print bed centered -110..110 (official machine profile), not 0..220.
Published assets: outputs/danya-simulator/assets/projection-prototype/ad5m-one-plate/
Default deliverable: DANYA-rear-head-AD5M-settings.3mf (two separate objects, AD5M 0.4 nozzle/PLA/.20mm/normal supports/by-layer/no brim). ZIP includes this, generic 3MF, combined+individual STLs, verification JSON, print images and existing v12 assembly PDF.
No assembly source files or master parts changed. Only projection-prototype.html got additive link to new guide page.
Rigid transform error < 0.000008 mm, same triangle counts and hashes verified. Thus optical/face/mirror interfaces unchanged.
Actual CLI slice of 2-object 3MF completed exit0. Headless shader thumbnail warnings but valid project exports. Final Gcode only retained in work, not shipped standalone. Estimates 15h23m01s / 481.47g PLA including supports. First STLs-only slice earlier 15h47m / 484g is superseded.
Final Gcode extrusion path XY with half line width .25 is within220 square: see slice-verification.json. Edge margin only ~1mm. Normal supports, not tree, no brim. First-layer/support footprints rendered from real Gcode. No machine printing performed. Surface support scars and support-removal/warping/strength remain physically untested; described explicitly in README and page.
Scripts: make-layout.py creates rigid files/standard3MF; profiles.py flattens installed Orca bundled profiles (no packages installed); render.py Blender views; check-gcode.py path envelope/images; publish.py validations/docs/package. Don't rerun obsolete builders for assembly geometry.
