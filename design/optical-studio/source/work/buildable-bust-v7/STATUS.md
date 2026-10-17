# v7 current state — 2026-09-23

User approved moving mass assumption2kg. Task: upper neck attachment, stock spring selection, physical strength test preparation, belt collision redesign.

Active UI outputs/danya-simulator/buildable-bust-study.html (iab tab12). Revised src/budget-system-study.js. Original head/face/body exterior unchanged.
Pitch pivot is world[pivot.x,pivot.y,-4], roll pivot[pivot.x,pivot.y,20], NOT a common pivot. New motion.mjs and UI hierarchy agree. Pitch direct258mmGT2/6; roll320mmGT2/6 with2idlers. Actual current motor positions/quaternions are drive-layout.json. direct-drive.mjs now preserves the selected roll motor as a literal.

CAD42validsingle solids:38installed+4fixtures. 27pose checks: CAD vshead/face/body/projector0; crossstageCAD0; alltransmissionenvelopes vsCAD/motors0; checkedtransmissionmutual0; motorboxes0; samplechief-rayobstruction0; displayedcollarsurface0. Head/face/projector vsbody0. PhysicaltestsNOT_RUN. RegressiontestsrigidOBBcentre and separatedpitch/rollpivots pass. UI extreme30/12/6 andresetchecked.

Review ZIP: outputs/DANYA-neck-v7-review.zip. Publicdetails andremaininggates inassets/buildable-bust-v7/DETAIL_DESIGN.md, VERIFICATION.md, MANIFEST.json. ThisisNOT manufacturingrelease.

Criticalremaining: pitch8mmshaft has27.5mmoverhang and~161MPacantileverscreen at3Nm, excludingpretension/crosshole. Needsteelgrade/heat-treatment or largeraxis/outboardbearing. Motor/idlerbrackets, actualpulleys/hubs/flanges, head/projectorcarrierattachment, wiring/thermal/power notfinished. Machiningcostwithin30000JPY notconfirmed. Donotpurchasethese asreleaseddesign. Physicaltestingcannotberunremotely; fixtures/procedureonly.

Spring SHOWA AP080-022-1.2, free22OD8wire1.2,k6.803. Install16-18mm, guide5.2x3.3x14.5,M3x40. Sliptarget0.08+/-0.01Nm, requiresmeasurement.

RebuildCAD: existingcad-venvpython build.py thenbuild-upper.py thenassembly.py. build.py includeslatestfeet x+/-12,y+/-41,M3x20tappedfoot,OD5L6standoffs; pitchaxisZ-4. CapsOD26PCD20angles90/210/330. CarrierOD86ID78Z37..43; bridgesZ35..41;negativebridgeinnerendX-36 clearslight. DoNOT restoreearlierv7candidateassets. OBBbugaudit documented; oldv4-v6motorpairclaimswithdrawn.

Run direct-drive.mjs thentransmission.mjs forcurrentdrive; final checks check-parts.mjs,check-assembly.mjs,check-motion.mjs,check-collar.mjs. Regenerate assembly.step afterCADchanges. package.py assertschecksandhashescurrentartifacts. No newdependenciesinstalled thisturn; userrequiresconfirmationbeforeinstalling.
