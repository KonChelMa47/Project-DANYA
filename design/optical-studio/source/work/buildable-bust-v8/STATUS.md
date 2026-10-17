# v8 status — 2026-09-23
USER JUST CONFIRMED: 3D printing + commercially available hardware only. No custom metal machining/bending/welding fabrication. Budget motors/mirrors/etc JPY30000 persists. Must redesign v8 metallic fabricated supports and existing v7 machined carriers into buildable printed/standard-hardware design. DO NOT call v8 finished or manufacture-ready.

Latest user: 「3Dプリントと市販金具だけで組み立てる方式にしますか？ はい、これでお願いします」.
Reply already sent acknowledging printed + stock hardware redesign, eliminating cut/weld-required mounts.

v8 exploratory metal mounting CAD has been built, not accepted fabrication approach. 132 valid single-solid parts (128 installed +4 fixtures). UI JS now reads assets/buildable-bust-v8, HTML title fixed mounts v8, note says revalidation/not release. User tab still may show old cached v7 until reload. Existing v7 archive preserved.

Latest checks frozen after rebuild.py:
- device-check.json:128 rows, NO contacts vs actual motor STLs, belts and provisional pulley envelopes (explicit shaft/pulley and horn mating exclusions).
- assembly-check.json:4 cross-stage pairs, all enlarged motor-yaw-saddle vs folding mirror adjuster0/seat0/adjustmentplate/leftopticsframe in1-2poses. Must fix, not passed.
- parts-check.json: completed. Uses official TI mesh now instead of conservative hull. Inspect rows; pads contacting projector intended. All sample light checks look clear, confirm report.
- same-stage non-mating contacts NOT yet audited; known base-right foot and yaw-motor-support foot centers10mm apart withR6 footprints overlap2mm. Must redesign/remove or shrink foot radius4.75. Current parts remainR6.
- No physical strength tests. assumed moving mass2kg user approved. No budget quotes or purchases. Actual pulley hubs/flanges STILL provisional; not resolved by adding mounts.

Rebuild sequence work/buildable-bust-v8/rebuild.py uses OCP python work/internal-projector/cad-venv/bin/python, copies v7 base CAD, runs build-mounts.py, build-drive-supports.py, build-frame.py, build-fasteners.py, build-interfaces.py, assembly.py. Outputs assets/buildable-bust-v8. All scripts executable with current environment, no new dependencies installed. Node path /Users/daniilmalchenko/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.

Added v8:
- Servo split saddles actual CAD case envelope corrected: localX12..24, Y-43.5..-1.7, Z+-18.4; pocket Y-38.3..-1.6,Z+-12.9. capY-1.7..1.3, two M3x45 atX18,Z+-15.65. rear bracket holesX14.5/21.5,Y-43.5,Z0; M3x8 flatheads fromY-46.5. Saddles6061 proposals -> must redesign printed now.
- yaw motor turned -90deg around output axis, shaft/gear centers unchanged. Current drive-layout.json has final quaternion. This avoids motor body hitting fold mirror; widened saddle still4 contacts. Pitch/roll motors unchanged from v7. Outputs not changed.
- Mirror backplates82%(main)/88%(fold) outline,3mm thick localZ-5.1..-2.1, three adhesive pads thickness1 at-2.1..-1.1. Reflective faceZ0 andglass1.1 unchanged. Fixed adjustment plateZ-11.1..-8.1 with3holes at(+/-0.25W,-0.18H),(0,.25H). flathead M3x12 starts-14.1 ends-2.1;3mm adjustment gap must not shrink (tip/glass clearance1mm). Main lower two heads sit in frame lugs, upper third3mm countersunk spacer. Fold all3 have3mm head spacers; independent central M3x6 fixes plate to frame, tappedholeat0,0. All currently custom aluminium -> printable redesign needed.
- Projector open tray based official kit coordinateX0..135,Y0..3.97,Z0..75, lensdatum[111.62,13.97,81.335833]. World registration projector layout rotation/translation after subtractdatum. TrayX2..133,Y-6..-3,Z2..73, two big windows. Rigid lower3mm aluminium pads atX12/123 Z7/68. Peripheral clips final coords(X,Z)=(16,-3),(48,-3),(80,-3),(96,78); old12/123edgeclip positions collided TI components. ClipsY4.47..7.47. soft upperpads supplied1mm, assembled0.5 fromY3.97..4.47 atinside edgeZ0..3 or72..75. spacersOD5.6 ID3.3 L7.47 outsidebaseedge. M3x20 inserted upward fromtrayY-6, heads belowtray, nuts aboveclip; reverse previous downward direction to clearfloor. Tray+clips customAl currently; can replace with printed dimensioned parts if tested, choose heat-resistant material and airflow.
- Additional pitch688 bearing movedinboard toY+56..61 (center58.5), existingbearingcenter42.5, pulley70 ->overhang11.5 vs27.5. Shaftremains8mmY34.5..75. Supportbridge/capmachinedAl. Addedlegbosses atlocalX+-12,Z4,Y39..45 withM3taps; capnotchesR3.25 atsamepositions; bridgefeetR3 atY45..48+uprightsX+-12,width4,Z4..16. newbosses connectrealmetal but needprinted alternative.
- Idler carrier two sides forroll,inner root atworldX4.5,Y36,Z10; path viaX5,Y59,Z-3 toaxisX5,Y60.3457,Z-10 thenY94.10395,Z-24. OuterrootX22.5,Y24,Z12; endsX22.5,Y60.3457,-10 and94.10395,-24. rootlugX3..6 inner and21..24 outer;holesworldY36(inner)/24(outer),Z8/14. yoke matchingM3pilots atworldX7/19.5. Outeridlercarrier merged into roll-motor-support metalframe to avoid overlappingrootlugs. Provisional5mm smooth axles X3..26, end retention not fully detailed. MR85 bearing assumption is not adequate/source-selected; current notes say provisional. Research found SKF PCM050708E plain bushing5x7x8 as REAL alternative (manufacturerPDF linked below), not implemented.
- Metal formedrod6 frame paths found by A* collision+light checks. Most supports needprinted/stock redesign now; DO NOT simply print6mm rods. Leftframe combines mainleft/projectorleft/fold support/negativeheadbranches; rightmirrorframe combinesmainright/positiveheadbranches. Joint graph assembly requiresfabricatedsteel welds (rejected user preference). Three servo rear steel plates fused with correspondingframe, two flathead countersinks preventhead/rod contact.
- Head inner shoes atZ55/70, radialdirection[-.45,+/-.89302855,0] fromcenter[-22.016,.3457,z]. Positions inhead-pads.json. Radius6 backing, innerface5mm behindwall,3.5mm thick,1.5mm adhesivegap, M3tap. Fouroriginalbackwardpads65/100 obstructedbeam ->replacedwiththese sidepads. Outsidehead/facegeometry unchanged. Good potential for PRINTED head-shell-supported mirror mounting instead of long fabricatedsteel rails.
- Floor plateellipsecenter(-7.67,.3457),rx80,ry109,Z-159..-155,4mmmetal;centre60x70opening. RodfeetZ-151..-148 on4mmspacers. Does not yet resolve manufacturing of originalbodyshell mountingbosses/holes. Exteriorbodyv3 untouched.

Routes chosen rebuild combines search-requests-result, search-more-result (exceptyaw), search-final-result, search-base3-result then replacements search-refine-result, search-pitch-root-result, search-main-last-result, search-base-last-result, search-base-top-result. Failed candidates NOT picked.
- Mainleft [-48,-29,44],[-53,-59,194],[-53,-54,209],[-59.73,-31.1,211.2]
- Mainright [-48,30,44],[-63,55,184],[-63,55,199],[-57.82,33.79,211.2]
- Projectorleft [-48,-29,44],[-33,-19,-51],[-38,-29,-76],[-33.44,-33.65,-78.42]
- Projectorright [4,-29,44],[-11,-59,4],[-11,-64,-1],[-16,-79,-91],[2.1,-78.17,-100.25]
- Fold [-38,-29,-76],[-33,1,-61],[-37.09,5.84,-51.52]
- Pitchsupport [-12.016,49.8457,-2],[-2.016,49.8457,3],[-2.016,24.8457,-12],[-32,17.3,-105.07]. root2M3atlocalX10,Z-5/+1 inlegandcap, frameflangeY46.8457..49.8457.
- Rollsupport [22,24,12],[62,59,-38],[67,75.59,-59] mergedwithouteridlercarrier.
- Yawmotorsupport [-72.2456,11.3457,-76.9],[-62.2456,56.3457,-86.9],[-62.2456,66.3457,-101.9],[-50,75,-148]
- Baseleft [-49,-46.42,-31],[-79,-46.42,-66],[-59,-71.42,-136],[-50,-85,-148]
- Baseright [-49,47.11,-31],[-49,52.11,-36],[-44,67.11,-86],[-44,77.11,-91],[-50,85,-148]
- Basethird NOW topmount upperretainer: [4.984,-46.4196,-7.7],[4.984,-56.4196,-7.7],[4.984,-106.4196,-57.7],[-.016,-101.4196,-137.7],[-10,-90,-148]. bottommount collidedprojector; topflangeZ-10.7..-7.7 needs longerretainerbolt.

Verification utilities:
- route-check.mjs exportsobs, check,checkRod,pointTester,cylinder,stage. obs includesinstalledCAD,actualmotors,actualTIkit,head/face/body,beltSTLs,provisionalpulleys. check() checks27poses; canignorebynames; chief-rays union currentfaceevery200vertices AND work/torso-flat/face-local-v2 every200. No finitepupil.
- find-routes.mjs A*5mmvoxel pointTesterR3.6, pruningcylinderR3.6, finalrodsR3. InitiallypruningR3polygon12underestimatedOCPnearbeam; increasedsafetymargin.
- check-parts.mjs actualTIkit instead ofoldconservativehull;every200 face-local-v2rays,27poses. Expectedcontactprojector-base-pad-* andprojector-edge-pad-* withprojector must be explicit.
- check-assembly.mjs copiedv7, crossesstagepairs only, shaftnominalfitexcluded; DOES NOT prove same-stagejoint non-interference or containment.
- check-devices.mjs128 installedparts actualmotor,belts,provisionalpulleys, contactexclusionsspecificshaft/pulley+horn.
- Need same-stagecontactaudit, continuousmotion, expandedactualpulleys,wire/ports, andmanufacturingbudget. No verified CAD claim beyond scope.

UI: outputs/danya-simulator/src/budget-system-study.js now referencesv8. HTML updatedheadline,note,sectionpending; mostdownloadlinks stillv7 andmustfixifshippingv8. v8 VERIFICATION.md overwrittenwithdraft(notstalev7pass). Otherdocs inassetv8 werecopiedv7andareSTALE; rewrite ordo notpublish/archive them. No v8ZIP createdyet. No final user responseyet.
Browsertool: cua initializedgetState, bustTab=await cua.getTab('12',{browser:'2'}). ItreturnedAXstateonly, no furtherAPI docs. Do not invent undocumented APIs. Can documentedcreateBrowserTab/getTab/getState. Browsercurrentlyshowscachedv7 despitefilesv8. No screenshotnewviewyet.

Dependencies: no installs. User AGENTS requiresconfirmationbeforeANYinstall. No subagents unlessexplicitrequest (none). ExistingOCP/Blender/Three/BVH suffice. Need continue untilnewprinted/stockhardware redesign meetsuserintent, not stopwithv8metalproposal.

Web primary sources openedthisturn:
https://www.ti.com/tool/DLPDLCR3010EVM-G2
SKF PCM050708E official5x7x8 bushingcatalog:
https://cdn.skfmediahub.skf.com/api/public/0901d19680090e01/pdf_preview_medium/0901d19680090e01_pdf_preview_medium.pdf
https://cdn.skfmediahub.skf.com/api/public/0901d19680229dfc/pdf_preview_medium/0901d19680229dfc_pdf_preview_medium.pdf
Only researched, not implemented/procured. Do not quote bearingloadratingswithoutreadingtableheaders.

Goal tool get_goal showed olderopticsgoal BLOCKED (3milliontokensused) notcurrentmounttask. Do not markcomplete/reopen/createinferredgoal. Usernewtask active throughconversation.
