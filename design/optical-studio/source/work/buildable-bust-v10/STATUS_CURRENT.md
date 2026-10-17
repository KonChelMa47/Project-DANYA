# Project DANYA — active interference redesign, 2026-09-23

INCOMPLETE: do not fabricate or describe the entire assembly as collision-free.

Current isolated viewer: http://127.0.0.1:8769/revised-clearance-study.html
Source manifests: revised-structural-candidate.json / revised-drive-layout.json / revised-transmission.json under outputs/danya-simulator/assets/buildable-bust-v10.

## Current geometry
- Original head/face exterior preserved; head-inner-relief changes only557 inside vertices, minimum normal wall0.7988mm. Face/head nominal seam offset0.
- Four magnet pairs Ø5.9×2.9, pocketØ6.15, magnets inset0.15each, carrier facegap0.2, magnetgap0.5 so magnets cannot prop exterior seam open. Physical fit/retention untested; thin opaque returns still cast edge shadows ~0.3% in earlier optical screen.
- Main mirror cradle inside head; side rails end89.226mm below mirror glass. Same rigid stage; this separation is invariant.
- Upper mechanism+8mm, projector/first mirror additional−6mm. Actual TI kit,135×75 measured base, ST3215 motors retained.
- Roll motor housing rotates20° about unchanged output datum, revised300mm belt path/idler.
- Shoulder300mm; upper-neck opening136×154mm. Body bottom extended4.5mm, total188.5mm. Integral floor bosses in body-with-floor-bosses.
- Latest floor update: removed front-center boss at(60,0), replaced by(20,−85) and(20,+85); seven M4 floor fasteners total. Shell5bosses, rear feet2. Latest body mesh is closed single component.
- New optical support routes revised-projector-support-right/left; revised-roll-inner/outer are STILL thin metal rod envelopes, not all-printable structures. Print+stock conversion unfinished.
- Pitch support route corrected after original detour hit yaw ring. Latest builder uses pitch-core-detours.json first passing route, subtract8 before CAD export.
- Lift hardware: four CU-31414mm spacers + M3×30 countersunk bolt envelopes, liftExempt:true. Legacy machined legs still violate all-print/stock constraint.

## Validation as of this checkpoint
- check-revised-snapshot.mjs ran525poses(yaw±30,pitch0..12,roll±6 at3°), valid input hash. Major motors, mirrors, supports had only mating contacts after pitchcorefix. Report became stale when floor/tray updated later.
- revised-shell-magnet-screen.json checked all16magnet parts: no surface hits. It exposed tray/pitchbackplate and tray/frontfloorboss contacts.
- revised-mating-volumes.json: listed exact STEP intersections all≤0.0001mm³. Not all interfaces or current relocated floor details.
- continuous-revised-clearance.json RUNNING from previous frozen geometry: important motor/support pairs passed; latest pitch support vs base lower bound1.5433mm and sleeve1.0078mm. Right optical support/base calculation slow. Source body/floor has changed meanwhile, so recheck affected body pairs before final report.
- check-revised-all.mjs RUNNING, session24616: all modeled pairs, symmetric pairs once,525poses, hashes. Main tray relief still contacts pitchbackplate/washer; no floor/body contact after relocated bosses so far.
- Current tray has underside relief X40..72,Y−8.2..−3.05,Z74.5..80.5 in TI base coords. New separate revised-tray-relief-candidate (probe-tray-relief.py) extends relief toZ67.5, preserving full locating rim and135×75 base. NOT yet adopted/checked.
- User browser tab16 displays revised comparison; AX page loads and source status says face plain because UV update pending.

## Required next work
1. Check separate deeper tray relief, adopt only if all relevant contacts clear; don't mutate current hashed assembly until all-pair screen ends or explicitly invalidate run.
2. Inspect ALL all-pair results, classify real contacts versus mates/thread envelopes; correct remaining interference.
3. Continuous-angle verification across allowed range, not only discrete525poses; rerun affected body pairs after floor changes.
4. Preserve measured magnet geometry and seam, verify physical retention/fit later. No physical tests performed.
5. Full projection shadows/UV/focus not solved; no100% optics claim. Print+stock-only detailed structural fabrication and hardware access unfinished.
6. Update UI text and reports before final delivery. Do not label manufacture-ready.

No dependencies installed, subagents, purchases, or physical tests. Moving mass2kg assumed; budget≤¥100000. Do not silently narrow motion ranges or change head shape.

## Superseding update, later 2026-09-23
- All242parts completed `check-revised-all.mjs`, validSnapshottrue. Subsequent only-tray and only-support changes were incrementally rechecked with tracked hashes. **Latest authoritative sample result: revised-current-screen.json**, validSnapshottrue, finished2026-09-22T22:49:03.178Z. Current geometry hash audit reports no changes after that run.
- Tray underside relief adopted: X40..72,Y−8.2..−3.05,Z67.5..80.5. Clears pitchbackplate/washer;135×75 locating wall preserved.
- Real positive STEP overlaps found: rollouter/axle62.65mm³, opticalleft/tray88.74mm³, opticalright/tray137.82mm³. All FIXED. `repair-support-seats.py` now recuts rollouter axleholes after branch union, and subtracts actual tray solid from optical supports. Do NOT use earlier broad plane cuts/gussets: those disconnected the part and were rejected. Final3seatfix shapes each1solid. Adopted tooriginalnames. `revised-seatfix-mating-volumes.json` checked overlap0 for all3 and other listed pairs.
- Rollouter/body had actual0.85977mm gap at yaw−30,pitch0. `bevel-roll-support.py` makes4mm corner bevel at motorlocalX12,Z10; same mount holes and axis. Adopted `revised-roll-outer-clearance` to`revised-roll-outer`. `continuous-roll-bevel-body.json` proves≥1mm against current body through whole permitted yaw/pitch range.
- `continuous-body-fine-spheres.json` proves≥1mm for right/left opticalsupport,pitchsupport,rollinner vs currentbody. Old rollouter unresolved row is superseded by bevel report. Coarse `continuous-body-spheres.json` is diagnostic, not final.
- `pinion-hub-continuous-envelope.json`: height-sliced triangle rotational outer envelopes prove hub/ring≥4.999994mm for ANYspin/yaw. Wholecylinder collision was conservative falsepositive. `check-pinion-hub.mjs` actualgeometry sample also nohits.
- `mirror-cradle-seat-plane.json`: mirror-local mating planes separate cradle/fixedplate withinFloat32−0.0000269mm noise. STEP cradle unavailable; this replaces exactvolumecheck for that interface.
- Old `certify-revised-critical.mjs` process2070 was intentionally terminated at slow right/body pair. Completed18namedpairs saved in `continuous-revised-clearance.json`; all18passed≥1mm. It used pre-seatfix/pre-bevel supports and previousbody; support material removal preserves their clearances, but body proofs are replaced by current fine-sphere reports.
- `certify-left-support.mjs` coarse cover unresolved vs motor-yaw/base/sleeve; motor-pitch/roll passed. Finer angular-only attempt `continuous-left-yaw-refined.json` still unresolved (sphere approximation, not a proven collision).
- **RUNNING session18015**: `certify-left-fine.mjs` using triangle edge≤0.75mm, voxel0.5mm, per-sphere rotation radii for leftsupport vs yaw motor/base/sleeve. Log `continuous-left-fine.log`; output `continuous-left-fine.json`. Inspect and continue after completion. Covers current adopted leftsupport; no geometry mutations while running.
- Bundled node runtime disappeared midturn; load_workspace_dependencies reports noneconfigured. **No install.** Use existing VSCode embedded Node24.18: `ELECTRON_RUN_AS_NODE=1 '/Applications/Visual Studio Code.app/Contents/MacOS/Code' script.mjs`. OCP Python and Blender stillwork.
- `INTERFERENCE_REVIEW.md` underassetsv10 is user-readable Japanese change/verification report. Viewer now links it and authoritativecurrentreport; magnets shownbydefault. Oldoriginalpage has latestv10link. `open_in_codex` queued latestrevisionURL but not confirmednavigation. HTTP200 all3page/script/report and JSsyntaxpassed.
- Remaining: full whole-assembly continuous-angle/containment/deformation/cabling certification NOTdone; thinmetalrod fabrication conversion, endpointretention andrealtests NOTdone; magnetreturn opticaledgeshadows NOTfixed. Do not claim finishedprintready or all-angle guarantee.

## Latest adopted central mirror approach — final checkpoint in this segment
- Old left support closer to yaw motor at intermediate poses: sampled exact gap1.45088mm at yaw−30,pitch12,roll0.23584. Fine spheres/primitive hull proofs were unresolved near1mm; these diagnostic failures are NOT reported as actual collisions.
- Replaced ONLY the fold-mirror branch with `[[-38,-29,-74],[-30,-10,-66],[-36.39830389097089,4.657758158453259,-64.89600337676927],[-39.032278487186446,9.21993598494433,-51.771647723787126]]` liftedworld coordinates. Newentry is foldlocal[0,−14,−20.6]; mirrorpose unchanged. `fold-alternate-probes.json` first3-checkpassingrow. Earlier downward/sideward branch probes collided and were NOTadopted.
- `build-central-optical-support.py` exported1validsolid; tray removed by exact Boolean, so mating seats preserved. Adopted`revised-projector-support-left-central` ascurrent`revised-projector-support-left` STEP/STL. Previousasset backedup`*-before-central-entry`.
- `central-support-mating-volumes.json`: carrier/tray/foldplate/twoframewashers/twoframebolts all0mm³.
- `central-support-screen.json`:525poses+sampled chief rays; mating-only hits, lightHits0.
- `continuous-left-central-motor.json`: CAD capsule/boxcover against yaw motorconvexhull planes, allcontinuous permittedangles,≥1mm.839cells.
- `continuous-left-central-core.json`: enclosing annulus from actualbase/sleeve triangles; yawinvariance removesyawdimension. Both≥1mm,219/473cells. Envelopeinner35.98288576mm includeschorderror; sourcefixedmeshes contained.
- `continuous-central-motors.json`: currentleft vs pitch/roll motors≥1mmcontinuous.
- `continuous-central-body.json`: currentleft vs currentbody≥1mmcontinuous.
- `continuous-motor-body.json`: all3motorconvexhulls vs body≥1mmcontinuous(stage0yawfixed,stage1yawonly,stage2yaw/pitch).
- `continuous-head-body-height.json`: exacttrigvertexmin overpitch/roll; head/bodyverticalgap≥11.9750059mm,face/body≥15.9023161mm; yawinvariant. Neckcoverexcluded.
- **Authoritative `revised-current-screen.json` now finished2026-09-22T23:16:21.624Z**,242parts,validSnapshottrue. Latesthashaudit showed nochangedinputs. Incrementalcentralsupportdelta recheckedchangedpartagainstALLobstacles, unchangedpairs inherited. No geometry edits afterthisrun.
- `build-revised-optical-supports.py` updated to usecentralentryandtrimactualtray; `build-revised-roll.py` updated toreamafterbranchunion,bevelcorner,andpreserveexistingtargetmanifest. Syntaxcompiled only, notrerun; currentassetsunchanged.
- Old long exact and fineleftjobs were stopped; NO activeCAD/proofjobs remain asofthischeckpoint.
- UpdatedJapanese`INTERFERENCE_REVIEW.md` andviewertext. All notes about unfinishedwholeassemblycontinuousproof, physicaltests,cables,flexiblecollar,print/stockfabrication,andopticaledge shadowsstillapply. Do not call thismanufacture-ready or all-angle certified for everycomponent.


## 2026-09-23: common head connectors v3 (static AND motorized)
User clarified both designs. Active rear head now has trimmed neck interior, flat bottomZ50.65 (0.54mm trim), integrated mirror cradle and four captive M3 nut bosses. Static arms and motorized upper head rails changed to bolt-on printed parts. Four M3x8 +4 M3 nuts +12 washers t0.5 added. Mirror cradle/4oldmetalshoes removed as separate production parts. V2 face/head magnets retained; production viewer now uses integrated v2 face and v3 head, prelift bins under assets/head-connectors-v3. Main work scripts in work/head-connectors-v3; read CONNECTIONS.md there in assets.
Checks:31 static printSTLs, all1connected0nonmanifold. Head1solid0nonmanifold.167698rays:staticnewarms/hardware0obstructions; head+cradle0before/after. Motorized roots retain existing510left/286right blocked samplecounts, newlyblocked0.525 discrete poses for changedparts:no unexplained contacts, intentionalfaces/threadenvelopes excluded. Matingvolumes0 with0.01mm separation for exactcoplanar support/mirrorfixedplate contacts. Physicalstrength, wholemechanism continuousangles, fullface/thermal/focus still unqualified. StaticZIPupdated, oldv2 archive-before-head-connectors-v3 preserved. Do not regenerate with oldbuilders blindly.
