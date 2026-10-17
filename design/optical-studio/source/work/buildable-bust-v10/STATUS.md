# Active task checkpoint — 2026-09-23
User asks fix mirror/left-right posts, add5.9diam x2.9thick strong magnets for gapless face/head joint, eliminate collisions at ALL yaw/pitch/roll positions. Existing authorized ranges yaw+-30,pitch0..12,roll+-6, preserve external head/face.2kg assumed,100kbudget,printed+stock hardware. No subagents.

## Pending user question
Async install approval requested for manifold3d +numpy in work/internal-projector/cad-venv using absolute python -m pip install manifold3d numpy. Required by user AGENTS before installs. NOT authorized yet, DO NOT INSTALL until reply. Reason: Blender boolean head rebates generates invalid nonmanifold mesh. Continue independent mechanism work.

## Findings / verified changes
- audit-visible.mjs loaded v8 visible frames +ACTUAL glass (previous checker omitted glass!). Reproduced mirror vsboth optics rails; motor-pitch vsframe-yaw-motor-support at yaw3deg; motor-yaw vsleftopticsframe at -30,12,3.
- mirror-head-cradle.stl: printable bridge +2conformal14x24mm adhesivefeet onheadinterior. Replaceslonguppermirrorrails. twoM3x12 attaching existingv9printedmirrorfixedplate. CADbridgeSTEP, fullconformalcradleSTL only. Manifold0edges,volume16697.5. No sampledlight/contact failures exceptintendedfixedplate/bolts. Adhesives/physicalstrength untested.
- v10backplates corrected an additionalmissed collision: priorv9framebolts entered CENTRAL servo back atY-38.2. NewmainplateY-42.6..-38.6 thickness4, isolatedpads toY-35.2. M2x10 through0.3washer (tips-32.9 intoactualrear pilot); threadspecstillunconfirmed. CaptiveM3nutsY-41.2..-38.8, frameM3x8 tips-38.6. M2touchholes intended. All33CADpartsvalid. Filemotor-backplates-v10.json.
- yaw-motor-rear-stand.stl: printed2leg stand, rootcanonical[20.5,-50,0] usingv9yawquaternion. Feet CURRENT[-60,-60,-148],[-40,-60,-148] (trialsY-85 hitbody, originalsecondY-30only0.15margin). RecessedheadM3x8 tobackplate, M4floorfeet. Motororientation remainsv9selected yaw-45,pitch+15,roll0 relv8.
- certify-motion.mjs --critical COMPLETED currentstand+mirrorcradle vs3motors+body: all8pairs have continuous triangle-surface separation >=1mm withinrange, using conservative motor convex hulls +interval rotationaldisplacement bounds. Notfullassembly, initialcontainment notproven. continuous-critical-clearance.json currently8rows, notallpartsproof. Yawmotor-stand4.8mm, body18.44mm, stand-pitchguaranteed1mm. Otherparts stillmustcheck. Fullwithoutcritical slowforactualTI, nowusesprojectorhull too.

## Full structural candidate / remaining collisions
assemble-structural-candidate.py ->168parts structural-candidate.json. Copiesv9retainedcore/modules, v10backplates/stand/cradle, legacyv8frame-base*,floor,pitch/rollsupports,headshoes; trimsframe-optics-left/rightatZ85 sohead stillsupportedby55/70shoes buttallmirrorrailsremoved. Severallegacyframe endpoints DO NOT yetmatchnewmotor/projector/mirrorinterfaces; NOT RELEASE. No magnets loaded. Legacycore/framesstillcustommetalmanufacturing, not allprintable.
full-route-check.mjs loadsALLcandidate168 +actualdevices/glass/body/head;3degreeangulargrid, cachedrelativeposes, convexhullfastrejectonly thenactualintersection. check-full-frames.mjs completed/nearcomplete: intendedjointcontactsplusreal issues:
- frame-projector-right vsframe-base-left/third andnewtray
- frame-base-right vs pitchbelt/pulley
- frame-base-left vs newtray
- frame-base-third vsprojector/newtray/pitchshaft
- floor vsprojectortray andclipbolt atsomeposes
- oldpitchsupport vsnewtray (bridge/capintended)
- oldrollsupport vsnewbackplate (needsendpointadapter)
- shortenedleftopticsframe vsnewtray+foldmirrorfixedplate/hardware (needsadapter)
- shortenedright opticsframe onlyintendedcarrier/headshoe contacts.
Do NOT callassemblycollisionfree orcomplete.

## Promising next: replace long fixed baselegs withrearbodycage
probe-back-chassis.mjs ->back-chassis-probes.json. TwoR5.6curvedspines followingbodyrearinner atY+-60 or+-70, centerX=innerrearwall+7, Z-148..-28 every15mm; ALLsegments atY+-60/+70 pass27poseactualdevices/core/body/chief-ray check. User-approvedbodyshapeunchanged. Can printR5beams linkedbytopcrossbar, shortarms to fixedyawbearing-base, avoidslonglegs crossingmotors/belt/projector.
probe-back-arms.mjs JUST STARTED: sixconnections frombearingears[-49,+/-46,-31] tospineZ-43; centerbackbearinghole[-76.016,.3457,-31] tobackcrossbaratZ-28; bottomfeet[-67,+/-60,-150]. Fileback-arms-probes.json pending.
Floorcollision plan NOT IMPLEMENTED: lowerfloor~6mm andadddiscreetdarkbasefoot, preservingheadandbodymainshape, toclearnewtraythroughpitchrange. ExistingfloorZ-159..-155; proposedprintedfloor6mmZ-167..-161, existingstandspacers8mm instead2; rearcagefeetwouldneed6mmspacers. Wouldadd~7mmoverallheight. Mustmodel/testactualscrews/outerbase, notsilentlymovefloor.

## Magnets
First6pairsatseam90/150/210 into12mmpads rejected: castshadows, somecontact. build-shell-magnets.py nowexportsSEPARATEconformalcartridges andUNCHANGEDoriginalhead/face (unionfailedpreviouslyanddestroyedshell, corrected). Do notadoptfirstcartridges.
Originalseamf(y,z)=1.1*work/seam_fit.npz polynomial(y/1.1,z/1.1); outerjointalreadyzerooffset. Originalhead222918verts445804tris,volume76380.17,nonmanifold0;face167698/335392,58238.27,nonmanifold0. Bothoutershapespreserved.
Rear4pairs candidates atZ~138/162,eachside;magcentersX~-18,Y~+-70, axisX closesfacerearwards. Magnetdiam5.9thick2.9,pockets6.15; bodygap0.2,magfacegap0.5 (recess0.15each), SHELLSEAM0. Face-returntongues bondoriginalcap with0.2gap, radialband1..5mm fromouterseam,zheight12,x=f-1.4..f-.2. Headrebate tools x=f-1.65..f+.15, band.8..5.2,height12.4. Rearheadcupsconformalbondfeet0.2mmgap.
build-rear-magnets.py nowSKIPSheadbooleans butexports4cutters+16valid/manifoldparts; rear-magnet-mounts.json headRebateValidFALSE. HEADNOTMODIFIED/released. OldBooleanuse_selfTrue produced1957nonmanifoldedges/volumeincrease; killedownBlenderPID89245. Do notuseoldpartialcutmesh. Newrear-magnets-candidate.blend forreviewonly.
rear-magnet-check.log:4headcups nohits/light;4faceturntabs noSURFACEhitsbutalllightHits1. diagnose-return-light.mjs correctlyusesuniqueouterfaceindices,1713samplepaths,baselineheadblocks0; newopaqueRETURN tabsblock3/4/6/4paths (0.2-0.35%) atfaceedges. Realadditionalshadow, NOT justfalseinner-targets; currentreturntabs mustredesign or opticalmaterial/physicalcheck. Do notclaim100%.
seam-cap.json showsfaceoriginalcapmaterialextendsseamX forradialoffset0.1..4mm; first-to-secondXthickness~0.2..3.6. Couldbond toexistingcap, butopaque finitewidthflange behindouterrimcastsedge shadow. RearmagnetCUPS themselvesclear. Needtangent/edgeattachmentortranslucentreturntab, withoutsilentlyrelaxingfullfacecriterion.
Headrebate maynotactuallyneeded: returntabs surfacecheckdoesnotintersecthead. Containmentmusttest (notdone); ifinsideexistingcavitycanavoidbooleans/install. No installapprovalarrivedyet.

## Runtime
OCPpython work/internal-projector/cad-venv/bin/python (no numpy/manifold/trimesh); numpyenv /Users/daniilmalchenko/Documents/Project-DANYA/.venv/bin/python; Blender4.5.1 --factory-startup; node ~/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node. Noinstalls/purchases. MainUIstillv8; projector-mount-study separatev9. Nothingv10shownyet.
KeepmeaningfulJapaneseupdates<=60s; notclaimfinished. Needcontinuework. No goals/subagents.

## Late updates after checkpoint
Reviewpage mount-clearance-study.html created (open_in_codex queued),168partstructuralcandidate actualTIprojector, oldcollidingframescoloredred, clearlyNOTcomplete.8criticalpairproofonly. Mainv8notreplaced. JSsyntaxpassed/all168STLexist. Page notyetbrowserQA.
Thin face return straps trial build-thin-returns.py:0.4mm thickribbon followsheadinnerwall+0.05mm (insidewall),16mmZwide; flange0.6mmthickX[-.8,-.2] relativecurvedseam, radialoffset.2..8,height20, followsseamY(z) interpolated. All4 thin returns samplelightHits0 butintersecthead (needclearance/relief). Builtall; testfulloutertargetsnotyetdone. Needcomponentcount/outerpreservation/headwallminthicknessverification. Originalrearreturnstabs stillcastshadows; notadopted.
CURRENT build-clear-returns.py triesribboncenterheadinner-0.15mm, flangeX[-.4,-.1],radialoffset.4..1.0; endingribbonX=edge-.25,inside1mmrim. Prefixmagnet-clear-. Thismayfitexistingheadbevelwithoutanyheadmodification/install; musttest. Ifnoheadcutneededcanretractinstallrequest (notyet).
RearbodycageR5.6spinesY+-60 clear butconnectionsdifficult. back-arms-probes: negativebasearm[-49,-46,-31]→[-96,-60,-43] clear;positivebasearmhitsyawingpitchbelt;straighttopcrossatZ-28hitsyawmotor. back-upper-probes onlyuppernegativeportangle240[-49,-46,-7.7]→[-82,-60,-10] clear;otherstraightarmscrosscore/motor/body. back-negative-probes curvedrearbodycrossbarZ-43clearnegativeY, yawmotorcollisionsatY0..30. rearcentralarm[-82,0,-35]→[-105,0,-43]hitsyawmotor. Frontnegativeporttoward[5,-80,-40]andbackspinehitsprojector.
Potential alternative NOT IMPLEMENTED: yawhousingrotation120relv8 (knownv9backplate-searchpasses27poses) insteadcurrent-45 wouldputbodyforwardX-55, freeingrearcage. Mustrebuilditsmount/standandcontinuouscheck, notadoptwithoutproof. Couldmountmotorbackplateto rearcageviaarmrootnew[~ -54,40,-80] to[-105,0,-73] above pitchingmotor insteadfloorlegs; trialonly. Two/threebearingbasemountarms needdetail, allstockhardware/printstrongbeamgoal remains.
No userinstallreply yet. No dependenciesinstalled.
