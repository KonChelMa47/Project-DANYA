# Active compact redesign goal — 2026-09-22

User explicitly requested continuing a loop until compact complete-projector storage and full-face illumination coexist. `create_goal` has been called; goal is ACTIVE, no token budget. Do not mark complete on sampled pinhole coverage alone. No dependency installation or agents authorized. User's original exterior and measured complete projector remain unchanged. Allowed 110% uniform head scale, interior changes and neck opening; two separate shells.

## Current work and immediate next action

`evaluate-curved-three.mjs` is screening **88,702** curved-main-mirror + TWO early flat-mirror arrangements (three mirrors total). It runs under exec session **73998** (may already be finished; inspect log/file). Log `curved-three.log`, output `curved-three-results.json`. By entry 45,000, 606 physically fitting coarse candidates exist; some R75 mm candidates achieve all 73 samples. This is preliminary only.

Once it finishes, run `curved-three-dense.mjs`: dense 2 mm front grid, 5,837 points, regenerated finite apertures and full body/outer-shell/mirror collision and ray tests for top 30. It was prepared but not yet started when this note was written. Caution: top results favor high R and may lack target diversity; if all share a failed geometry, rescreen compact seed data with diversity across main mirror / virtual source, or a specific smaller radius. Do not infer impossibility from ranking truncation.

Then adapt `local-vertices.mjs` to test all 85,642 exterior vertices, followed by triangle centroids, actual official body, thick original shells, finite pupils, nonsequential arbitrary mirror routes and focus. Use `focus-audit.mjs` and `pupil-coverage.mjs` as explained below. No new candidate has been installed into the public simulator yet.

## Scripts and models

Node: `/Users/daniilmalchenko/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`.
NumPy/matplotlib Python: `/Users/daniilmalchenko/Documents/Project-DANYA/.venv/bin/python`.
Run from repository workspace cwd. The existing app dependencies provide Three.js and mesh BVH. No installs.

`../adult-kit/load-model.mjs` registers full Tripo geometry and full official CAD. **In work scripts only, geometry key `ti-engine-rd` denotes the COMPLETE official kit**; public app has a different meaning for that legacy key. Never import an audit state with that key into app without remapping to complete-kit / Tripo geometry. App outputs still use previous explicitly unverified R16.5 layout.

All new searches initially use the 110% ORIGINAL OUTER head/face meshes in `work/adult-kit/original-*.positions.bin` with exterior-only indices. These omit wall thickness intentionally as a first feasibility stage. Full source meshes in `outputs/danya-simulator/assets` are from `robot_shell_clean_interior.blend` (panel removal confirmed); whole-shell check remains essential.

## New results so far

1. Diagnosis of previous three-flat failures: chiefly projector-body occlusion of incoming/final beams, not neck opening. Behind the previous large primary mirror, 25,920 conservative box samples gave no fit; this is bounded/conservative, not an impossibility proof.
2. `virtual-targets.mjs`, `virtual-below.mjs`: independently searched flat mirror / virtual source layouts. 15,360 + 9,600 trials, 31 + 3 unique usable mirror/source targets. Realization with 608 head poses and later 293 extra rear poses did not produce a valid all-face compact flat solution.
3. `rear-source-poses.mjs`: 1,620 pose trials, 293 full Tripo + official body fits against outer head/face. Additional rear positions reach lens X=-50; previous sampled poses stopped at about -38.5 for the common orientation.
4. Weak convex searches: `weak-curved`, `weak-refine`, `rear-curved`, `local-curved`. Curvature radii explored 25–1000, rather than assuming R16.5. Coarse success R60 and R35, but dense tests reveal remaining occlusion.
5. Current best **two-mirror** reference: `local-vertices-results.json`, first result. Full complete projector pose P[-40,-30,90], Euler[80,-10,-90]. One flat folding mirror and R35 primary at [10,0,210]. 85,393 / 85,642 exterior vertices reached; failures 102 self-face, 147 body. Dense 2 mm grid: 5,832 / 5,837 chief points. `vertex-obstructions.json` contains all 249 failing vertex coordinates.
6. `focus-audit.mjs`: forward finite-pupil sensitivity with hypothetical 4 mm circular pupil at mechanical lens rim. For the R35 reference, focus 160–170 mm gives median spatial RMS roughly 0.4 mm, p90 roughly 0.6–0.7 mm on 59 test fields; zero wholly lost fields in that sparse set. Earlier R40 reference optimum near 170 mm had median RMS 0.23 mm, p90 0.37 mm. These are NOT measured optics, and only outer surfaces were used.
7. `pupil-coverage.mjs`: reverse pupil visibility for white image, 32 area-distributed pupil samples, hypothetical D4/F165, official kit, outer shells. For R35 reference, 5,837 front-grid points: 3 zero-light, 164 partially visible pupil, 5,670 full pupil. Zero coordinates near nose: [81.424,-14,131], [83.696,14,131], [85.586,-12,133]. Source-pixel RMS median0.237/p900.660. Forward round trips to selected q agree ~1e-12 mm, checking reverse implementation. This is point sampling / conditional visibility, NOT calibrated irradiance or continuous-surface proof.
8. `pupil-refine.mjs`: 75 small main mirror shifts / R34–36 combinations on failure-biased 489 points. None eliminate every failed point. `guided-curved` targeted old failing vertices but regressed elsewhere: all-vertex result 83,983/85,642. Do not select that just because its weighted sample score improved.
9. `clearance-threshold.mjs`, `clearance.json`: R35 reference body and two mirrors have no found outer-shell separation below 1.5 mm with threshold-limited BVH search. This is NOT actual wall clearance after manufacturing and not exact global minimum. API returns a tested point even if farther than maxThreshold; compare returned distance to threshold, not return-object truthiness. `belowThreshold` was corrected accordingly. Original exact `clearance.mjs` was stopped because slow; don't rerun unnecessarily.

## Current three-mirror branch

`curved-targets.mjs` decouples final face illumination from routing around full body. 3,600 virtual optical configurations; 485 clear all 237 coarse face points; 451 primary curved mirror fits. Output `curved-virtual-targets.json`.

`curved-realize-seeds.mjs`: realizes the virtual source using TWO early plane reflections and 901 (not fully deduplicated) known fitting body poses. It uses analytic first-mirror distance t such that t+|P+t*d-B|=|V-B|, so the final curved mirror sees the same virtual source. Centre paths/body prefilter only at this stage. 88,702 candidate arrangements. Files:
- `curved-realization-seeds.json`: 430 MB verbose archive.
- `curved-seeds-compact.json`: 80 MB equivalent data without main outline (evaluator reconstructs apertures); prefer this.

`evaluate-curved-three.mjs`: 73-point early ray rejection, real Tripo body, outer shells, then reconstructed 4 mm padded hull apertures. Uses actual curved solid for primary and flat solids for first two; checks head/face/body both official and Tripo/mirror intersections and nonadjacent mirror beam blockage. Padding4 mm is NOT proof for any pupil diameter. Output top150 ranks high R; ranking can omit diversity. Started session73998.

## Next integration obligations if geometry succeeds

- Rebuild/check internal head and face thickness while maintaining original scaled exterior and seam. Existing head/face in `adult-kit-v1` were tailored to OLD R16.5 optical route. Do not assume valid for new route.
- Audit mirror/kit clearance and manufacturable mounting, and arbitrary bounce order, pupil clipping, focus spread. Don't call 100% throughput a requirement for 100% surface coverage: partly vignetted but illuminated patches must be reported separately from wholly unilluminated patches. No arbitrary visibility threshold may silently replace user goal.
- Actual exit pupil/lens prescription remain unknown. Measurement is only width167.5 × height94 mm at200 mm with focus possible, apparentTR1.194. Offset100 is nominal, not calibrated.
- Public source rendering currently draws raw video then multiplies the projection mask. For a new small footprint, this may crop video rather than warp complete face content into usable source pixels. Before declaring video goal complete, inspect/add an inverse face→projector warp with unique intended routing; do not merely mask away most video.
- Public `validateState` disallows outline on early folding mirrors. If integrating custom flat outlines, extend validation consistently (convex CCW bounds), preserve compatibility, and test meaningful invalid-shape cases.
- App reference layout/STLs not modified this turn. Public flat-study page is an earlier FAILED comparison, not current work.

## Latest continuation checkpoint (supersedes earlier 'running' notes)

All current searches finished; no solver should be running now. Goal remains ACTIVE; user asked to loop to achievement.

- Three-mirror screening finished: 88,702 evaluated; 3,767 pass early chief-ray screen; 2,122 physically fit coarse configurations; best73/73. `curved-three-results.json` retains150, with R75/R50/R35 and some diversity.
- `curved-three-dense.mjs` finished top30: best5,834/5,837. Best R50 primary at[0,0,215], body P[-30,-30,90], rotation[80,-10,-80]. Two early flat mirrors. Source body unchanged complete kit.
- `three-vertices-results.json`: 85,539/85,642 vertices, **all103 failures self-face on final leg** (`three-vertex-obstructions.json`), no body-ray obstruction in that all-vertex diagnostic. `three-pupil.json` D4/F165 still3 zero grid points, same nose positions as earlier; 149 partial,5685 full. `three-focus.json` optimum testedF170: median spatial RMS0.18mm,p900.34mm, sparse59 fields, hypothetical pupil/outer shell only.
- `three-pupil-refine.mjs`: 36 shifts of main only, D4/F170, failure-biased343 points. Best dx=-4,dz=-10,R45 has only2 zero samples (temple, NOT nose). All prior nose failures become illuminated under these assumptions. This initially kept old mirror outlines, so next aperture recomputation was required.
- `shifted-source.json` wraps that shifted candidate. `shifted-vertices.mjs` recomputed physical apertures from all85,642 vertices; main[−4,0,205],R45. All physical solids still fit outer shells/both full bodies. But chief coverage82,613/85,642: diagnostic `shifted-obstructions.json` shows **3028 body obstruction on leg3 (second fold→primary), ONE self-face vertex on final leg4**. Moving primary solved nose but moved beam into current body's upper edge. This is a useful optical target, not a solution.
- `inverse-fold-search.mjs` tried deriving single early plane and real body pose from fixed shifted optical virtual source:12,768 orientations,8 body fits,0 all73 ray fits. Do not regard as impossible; it is bounded and conservative-box filtered.

### Strong next direction

Fix the improved final curved-mirror / virtual-source optical field, and relocate the complete body with TWO early plane mirrors (three total), instead of retaining the current body's obstructing pose. The previous successful three-mirror realization machinery can do this.

First find final optical fields with zero self-face occlusion at **ALL85,642 exterior vertices**, not just237 or73. Current shifted field misses only ONE self-face vertex; try nearby primary positions/angles/R, or lower main e.g.z185–205 and R25–45 while keeping source virtual frame. Last-primary pose and incoming virtual source determine face visibility independently of how real body is folded into the head. Compute final-ray head/face obstruction only; ignore virtual source→mirror body intersections at this stage.

Extract virtual source frame from `shifted-vertices-results.json` first state by applying preMirror then foldMirror reflections to projector basis: p=reflectPoint, r=-reflect(r,n),u=reflect(u,n),f=reflect(f,n). Delete early mirrors in virtual state. Main changed position is NOT exactly on original center ray anymore; find actual center ray hit using `mirrorHit(new Ray(V,D), main)`. Use its distance as target.length when realizing. D=normalize(f+u*(1/TR*9/16/2)). Do not mistake primary vertex for actual central hit.

Once optical target clears face, reuse `curved-realize-seeds.mjs` and `evaluate-curved-three.mjs` against901 saved body poses, or broaden continuous poses. In seeds' center-ray body filter, use actual center hit instead of `last.position` for B→H if the last mirror moved off the original image-center ray. Dedup target keys must include R and last rotation, not only V,D. Full body rays/finite solids follow, then dense/vertex/centroid checks and finite-pupil/nonsequential/inner shells.

### Public files this turn

Added `outputs/danya-simulator/COMPACT_REDESIGN_PROGRESS.md` and links from index/README. It reports the unshifted R50 three-mirror stage (85,539/85,642;103 self-face;still NOT achieved). Main simulator reference layout and STL assets not overwritten. Keep progress current without calling preliminary results successful. No candidate state imported into main app.

## Latest continuation: BOTH-body all-vertex solution and internal-wall failure

Goal remains ACTIVE, not achieved, not blocked. No installation, no original STL overwrite, no default simulator candidate changed. Public COMPACT_REDESIGN_PROGRESS.md updated (see below). All processes finished as of checkpoint.

### Current strongest optical/mechanical candidate
- `margin-candidate.json`: full Tripo P[-34.5,-32.5,85.5], Euler[78,-8.5,-82], two early flats + primary R40 at[-4,0,205]. Keep work/app body-key gotcha in mind.
- `margin-routing-vertices.json`:4 tested,4physical, all85,642 clear in BOTH Tripo AND official complete bodies. Script includes finite solids/4mm footprint margin and mutual obstruction.
- `margin-clearance.json`: both bodies+all3mirrors pass1.5mm distance to outer head AND outer face. Threshold test, not global exact min.
- `margin-audit.json`:255,542 vertices+centroids:allfrustum,allouterTripo,allouterOfficial; original fullshellonly219,283. All25,5542 vs inner-body-main rays conditional on nominal pinpoint geometry. Coarse unmasked160x90 raster:1305intendedface,455alternate foldMirror>preMirror for both bodies. Hypothetical pupil D2,4,6 no alternate on103selectedfieldpixel samples;D8sixalternate foldMirror>preMirror>mirror. No double-projection clearance.
- `margin-pupil.json`: outerONLY reverse finitepupil D4,F165,32pupil samples×5837frontgrid:zero0,partial431,full5406,min.75,p05.96875; sourcePixelRMSmedian.50697,p90.87856. This is conditionalwhite visibility not actual irradiance nor actual lens spec. It uses work-key ti-engine-rd=fullkit; don't import as engine appstate.
- Earlier `dual-routing-candidate.json` alsoall255542 butonly1.278mm clearance and morealternate; superseded bymargin.
- Earlier `dual-routing-dense-tail` P[-30,-32,86] rotation78,-8,-80 passes1.5butfails10vertices. Not current.

### Search chain this continuation
`dual-routing-vertices.mjs/json` (8rows,6physical,allvertices) -> `dual-clearances` (best1.481mm fails1.5).
`dual-routing-dense-tail` evaluatesremaining64/114coarsescreen -> `dual-tail-clearances` -> `clearance-routing-vertices` (85,632 max).
`refine-margin-body.mjs`:729poses around[-34,-32,86]/[78,-8,-82], adds1.5mm officialhead distancefilter;184bodyfits151finalclear755seeds.
`evaluate-margin-routing`:755tested524coarsephysical.
`margin-routing-dense`:first30,20physical,5837all.
`margin-routing-vertices`:first4,all85642.

### Head walls: critical remaining issue
`wall-obstruction.json` olddualstate sampled25555:outer25555,headWall21909,faceWall25555,both21909. Face wall isn't cause.
`wall-legs.json` CURRENTmargin sampled25555:only finalleg4 blocked3646points. Early3legs clear originalheadwall.
`local-wall-cutters.mjs` CURRENTmargin all255542:blocked36259,90patches,blockingsegments worldbounds x7.254..55.839,y-71.175..71.626,z47.524..242.972.

### Unapplied cavity generation work (do NOT call successful)
`cavity-points.mjs STATE` saves4beam convex pointsets plusofficialbody,Tripo,3mirrors into `work/compact-loop/cavity-v2/`.
`cavity-v2/make_hulls.py` Blender hull+Minkowskicube padding. Defaultallpoints,`-- --walls-only` onlywallpatches. Beam/body/mirrorpadding.3mm;localwallpatchpadding.15mm,localblockedrayendpoints extend.3mm. These are chief-ray cutters, not finitebeam envelopes.
`cavity-check.mjs` builds BVH on cutters IMPORTANT (without it fullbeamcheck was veryslow; stoppedprocess). Checks headEXTERIOR only and1mm remainingdistance. Excludes rejectedbeam3 andoutputhead.
- `cavity-v2/check.log`:beam0/1/2,allbody+mirrorcutterspass1mm. WholelastbeamconvexhullINTERSECTS headexterior, rejected.
- `local-wall-cutters.mjs`:groups onlyactualfinalhead-intersections per10mmy/zfacecoordinatepatch,90patches.
- `cavity-v2/check-local.log` andcurrent`cutters-exterior.json`:68/90wallpatchcuttersintersectexterior;72fail1mm. Only18pass. They were NOT applied.
- `cavity-v2/apply_cutters.py` guardedboolean allcuttersmustpass1mm. NEVER RUN whilecurrentcheckfailed. Original fullhead input fromadultoriginalpositions + appassetsheadindices. Outputonlycavity-v2head+blend,notmainapp.
- Head/faceSTL mainassets unchanged. Might need veryfinepatches / clipping against protectedshell or differentfinalopticalfield. Don't bypassguard or eraseoutershape.

### Immediate next direction
Need understand why finalbeamheadwalls so severe and cannot trimwithout exteriorcontact. Possibilities broadconvexhull overcurvedopening vs thin/grazingclearance, matingrim/faceboundarydefinition. Inspect actualheadwallhits/exterior clearances and firstfailed opticaltargets instead of blindlyrunninghugebodies again. True finalrays clearouterhead,butmaygrazeboundary. Preserveexternalshape; canmodifyinteriorbutneedstructuralwall.
Two bounded finalfieldtests used FULLoriginalhead:
- `wall-aware-final.mjs/json`:192trial R30/35/40/45/50/60,dx0..35step5,dz0,-10,-20,10; movesprimaryonly,virtualsourcefixed.0coarsepass.
- `wall-aware-shifted.mjs/json`:same192,movesvirtualsourceandprimarytogether.0coarsepass.
Neither provesimpossible. They reject onfirst~172constraintvertices. Need loggingbestfailure counts/rayhitpositions, changevirtualcameraaim/mainorientation or thinwallnotfulloriginalwall, maybe optimizeagainstprotectedinneroffsetsurface.
Firstfaceoutervertex index271 q[17.136,-59.700,87.850]. This is sideface low, could be nearseam. Must investigate physically before excluding anytargets. Do not silently redefinefacecoverage.

### Public report
`outputs/danya-simulator/COMPACT_REDESIGN_PROGRESS.md` latest reportsall255542outer,1.5mmallparts,fullwall219283,finitepupilzero0,alternatepaths. It sayslocalizedwallcutters beingcreated; update to note90patchesfailed68exterior/72margin andnextinvestigation. Mainappstilloldreference,notnewcandidate.
Latestcommentary Japanese:小分けの加工案でも外形/肉厚抵触、未適用、厚い後頭部では未成立、遮光位置/接合部確認と主鏡再検討.

## CURRENT checkpoint — optical head, full face warp, published experimental v2

Latest USER STEERING: explicitly wants the complete FACE image, not a small cropped portion of screen. Implemented GPU full-face inverse warp and verified visible eyes/nose/mouth using real live source in browser. Goal still ACTIVE; no completion/blocked transition. This turn made substantial progress.

### New head solves inner-wall obstruction while preserving exterior
- `wall-diagnosis.mjs/json`:74blocked sampledtargets,0insideheadmaterial. Some targets exactly at outersharedseam; oldwallprotrusion causes shadows. Do not exclude face targets.
- `radial-shell/build.py`: buildsheadinneralongmainmirrorcenter[-4,0,205],normaloffset.6/.8/1mm,travelcap10mm. Exterior triangles unchanged. radialcenter only->~99%coverage.
- `head-ray-directions.mjs` -> `radial-shell/optical-directions.bin`: inverse reflection direction foreachoriginalouterheadvertexwhereprojectionPathsolves; fallbacktowardmaincenterotherwise.
- `radial-shell/build_optical.py`:inneralongactualopticalrays,nominal1mm normal,10mmcap. Fallbackwhenopticaldirectionnotinward. Inneronly. `.6/.8/1.0` variants.
- `radial-shell/split_fans.py`:split12bowtievertices into separatefansWITHOUT movingpositions. Prior naiveclosedheadhad12nonmanifoldverticaledges;fixednow.
- `radial-shell/bevel.py`: movesinnervertices atboundarytoward-X projectedintotangents,falloff4mmgeodesic; .15/.3/.6/1mmvariants. Allsample25555points clearbothbodyfit.
- Selected **bevel-0.6**. `radial-shell/bevel-0.6.positions.bin/indices.bin`222918verts445804tris.
- `radial-shell/topology-qa.json` on0.3variant:outside triangles EXACT equaloriginal,0openedges,0nonmanifoldedges.0.6hassametopology/outercoords but should recordQA0.6explicitlynexttime.
- `radial-shell/check_assembly.py` BlenderEXACTINTERSECTION new0.6headvsoriginalfullface:0vertices0faces0mm³materialoverlap (`assembly-overlap.json`). Bothpartsremainseparate. Exteriorunchangedat110%;originalfacefullshapeunchanged.
- `beveled-shell-audit.json`:all255542targets fullNEWhead+originalfullface+officialcompletebody clear. Tripoouter/officialouteralsoall255542.
- `beveled-pupil.json`:NEWfullhead+fullface D4,F165 conditional reversepupil visibility:5837points,zero0,partial431,full5406,min24/32. Identical to optimisticouterbodylimit. Forwardroundtripfirsthitoninnerfacemeans~filmthicknessawayfromoutertarget;don'tinterpretthatgap asinversefailure.
- `beveled-focus.json`:focus-bevel.mjs uses displacementrelativechiefFIRSTFACE MATERIALhit(correctedfromoutertargettoavoidfilmthicknessoffset);D4F160 bestp90;59fields,0zerohit0wrong,medianRMS.245085mm,p90.4230286. Hypotheticalpupil/materialideal,notrealcalibration.
- `margin-candidate.json`stillopticalpose;actualpublishstatechangeddesign/sourceasbelow. R40main[-4,0,205],P[-34.5,-32.5,85.5],Euler[78,-8.5,-82],fullTripo.

### Masks (chief only) and finite ghost issue
- `build-bevel-mask.mjs` official and `build-bevel-mask-tripo.mjs` Tripo:9nativepixelchief samples,fullhead/face. Outputs radial-shell/nominal-mask.bin/json andtripo-mask.bin/json. BOTH identical,148862enabledpixels,255542targetsnone lost.
- Boundsu[.3504783,.6850004],v[.2613057,.7303085]. ~429×338footprint. This is why raw videomasking waswrong.
- `package-bevel-mask.mjs` ANDtwo masks,copiesassetsnewghost-mask.bin/jsonwithcorrect maskSignature(publicgeometrytripo-projector/designcompact-optical-v2). difference0.
- `beveled-shell-audit.json` unmaskedfinitepupilD8 has7alternatefieldsample rays,routefoldMirror>preMirror>mirror. D<=6 sparsefieldinspectionnoalternate. UnmaskedchiefoutsideROI alternatefoldMirror>preMirror remainsbutdigitalmask rejects. NativepixelmaskNOTfinitepupilcalibrated.
- Need evaluate actual masked+WARPED source withfinitepupil, nativepixelboundarypadding/overscan, possiblebaffleorsafemask. Do not equateallreversewhitevisiblewithactualwarpfilledpixels illumination.

### Published experimental assets / source implementation
- `outputs/danya-simulator/assets/compact-optical-v2/`:head/face positions/indices/normals,face.outer.indices.bin,STLs,manifest,layout.json,ghost-mask.bin/json.
- `radial-shell/export_candidate.py` exportsfloat32exactexteriorandbinarySTL,normalscomputed. STLfilenamesDANYA-head-full-coverage.stl etc areexistingapppatternbutSTATUSexperimentalnotmfgrelease. Minactualwallthickness/uniformheat/mountingunknown;normaloffsetcapmakesminless1mm(~.426mmatvertexoldradial),don'tclaim1mmmin.
- `recommended-layout.json` NOW v2 experimental state(publicTripo,keycorrect),source.rotation270,display.viewoutside,modelScale1(adultcoordsalready110%). Previousreferencebackedup `work/compact-loop/recommended-before-optical-v2.json`.
- Appdefaultisnewcandidate. Query `?candidate=compact-optical-v2` alsoloadsassetlayoutandshowspermanentexperimentalcaution. Defaultheaderalreadycallsreferenceunverified.
- Appcachedmaskloadsnewcompact-optical-v2onallstartupnow. Otherimportedstateskeymismatchtriggersrecompute.
- `src/optics.js` validateState permitsnewdesign andflatconvexoutlinesforpre/fold. Allmirroroutlines nowvalidatedCCWconvex,bounds,etc.
- `src/app.js` updatesactualpre/foldrendergeometrieswithmirrorSurfaceGeometry/mirrorSolidGeometry whenoutlinechanges,insteadoflyingrectangles.
- `src/face-warp.js` NEW GPU inversefacewarp. PurebuildFaceWarpGeometry mapswholefrontalimageUV0..1acrossfaceYZbounds andmapsmeshpositionsintoprojectorUVNDC. Explicitface.outer.indices.bin fornewdesign;olderdesignsfallbackoutwardnormalXtriangles. Onlyalllittrianglesrender;maskappliedafterwarp. MeshBasicMaterialwithSRGBtextures/offscreenrenderer1280x720. Sourceorientationcanvasswapsdimensionson90/270rotation (NO letterboxorrotationalcrop),preservesfullframe. Fullframefitnotlandmark-calibratedyet.
- ApppaintSource replacesrawsourcecropwithfaceWarp outputthenmask. requestCalculation invalidateswarptopreventstaleoutput. calculate updateswarpafterlit/mask. LivevideojustupdatesGPUtexture,geometrycachedperpose.
- Nativeprojectorpreviewnecessarilylookscompressed/distorted;outside3Dfacehasfullfacialfeatures. Actualsourceisalreadyrenderedimageofpersonrotated90clockwise;270correctsupright.
- `work/compact-loop/source-frame.jpg` actualapi/frameusedforvisualinspection. Sourcefeedalive api/statussemanticreadytrue; originalsourcefilesweren'tedited.
- `index.html` explainswholefacewarpandprojectoroutputpreview;querycandidatebottomnoticefinitepupil/pinpoint/manufacturabilityunverified.
- tests21PASS (node --test outputs/danya-simulator/tests/*.test.js):addedface-warp.test.jsdemonstratesfullUV0..1mapsintosmallerprojectorfootprint;addedflatfoldconvexaperturevalidationtest. No depsinstalled.

### Actual browser verification this turn
CUA persistentvariables browser,tab(old4),candidateTab(new6). `cua.createBrowserTab('iab','http://127.0.0.1:8769/?candidate=compact-optical-v2',{visible:true})`openednewtab. `candidateTab.getScreenshot()` showednewcompletefacewithBOTHeyes,nose,mouthonexternal3Dhead,sourceplaying;UI100.0%,frontgrid5858/5858,200–313mm. (UIgridphase differs fromoffline5837;notcontradiction.) Scene3/4view,notfrontal. Headercandidatewarningpresent. Originaltab4remainsolduntilreload. open_in_codex returnedqueuedanddidnotnavigateoldtab;newCUAtab6isverifiedcandidate.
CUA getBrowser/getTab tool didnotprintAPI docs,but`nodeRepl.write(tab)` showedgetScreenshot/getAXState/click/etc. getScreenshot andemitImageworked. Don'tguessunseenmethodschemas. Canuseexplicitcua.createBrowserTab API ifneedrefreshbutavoidduplicates. Screenshotnativeisconversationimage,notfile. No screenshotartifactexportedyet.

### NEXT WORK (fullgoal stillincomplete)
1. Finishfinite-pupil+MASK+WARPactualemittedpixelcoverage/ghostchecks,notjustuniformwhiteinversefield. Consideredgepaddingandfinitemaskorphysicalbaffle. D8alternateexists.
2. Validatewarpactualtargetsourcefeaturelandmarksfrontalview andcoverageofrasterizedwarp(boundarytriangles/blackbackground). UserexplicitlywantsFULLFACEimageandcorrectface,notrawscreenslice. Implementedandvisuallydemonstratedbutprecisemappingeyenoselipstillcalibration.
3. Manufacturability/minthickness/selfintersections/clearto3mirrorsandbody+seamforNEWhead. Fullbody1.5mmouterdistancepassed;headbodyintersectionfalse. Needactualnewhead-to-mirrorchecksandmount/cable/heatallowances. Thinwallcapneck~.4mmcouldneedlocalizedreinforcementoutsidelightroutepermittedinterior.
4. PublicCOMPACT_REDESIGN_PROGRESS.md nowupdatedallnewresultsandlimits. READMEtopstilloldv1descriptionmayneedupdate. No claimgoalcomplete. Needspecsheet/render/exportBlenderfullassemblyoncevalidated.

## Latest checkpoint: diagonal aperture blade integrated (user asked status)
Previous goal turn progress; this turn ALSO progress. Goal ACTIVE. Latest user asked status; answered commentary: simulatedcompact/allfacecandidate,wholefacevideoworking,experimentalbladehelpsD8,realoptics/mountingnotconfirmed. All runs below finished; no live waits.

- `raster-warp.mjs`: CPU pixel-center triangle raster of current GPU full-face white warp, intersected native mask. `radial-shell/white-warp-mask.bin`95781enabledpixels. Approximation,notGPUreadback. Sourceimagefullboundsencodedintosmallerphysicalfootprint.
- `pupil-emitted.mjs` +`emitted-pupil-4.json`:D4F165 actualwhite-rasteractivepixels:5837targets,zero0,partial467,full5370,min10/32. Edge throughput lowerthanuniformwhite.
- `emitted-ghost.mjs/json`:officialbodyfullheadface,currentwhite-raster,pixelstride2,16pupils,383216rays perD4/D8. D4noalternate,D8alternate1644rays/allk12. These are BEFOREblade.
- `pupil-ghost-locations.mjs/json`:D8pixelstride4,64pupils,383104rays;1525alternatefromk33/41/46/49/54/59/62. Atlocalz1 theirx/yprojectedalong45degmax=-2.7549mm. Foundgoodknifeedgecutdistance2.4mm withmargin.
- Diagnostic pixel deletion:erase3pixaroundbadpixelsofstride2 ->8161pixelsremoved. `diagnostic-finite-mask.bin`, `pupil-finite-mask.mjs`, `finite-mask-coverage.json`:436zero/5837 -> REJECTED,NOTapplied.
- `baffle-study.mjs`:work-only finite opaque triangle inprojectorlocalframe:vertices(-6,-6),(6-2.4sqrt2,-6),(-6,6-2.4sqrt2),z.85..1.15. Absorbingcornerblade,notcirculariris. Plane1mmforward,lateralcutdistance2.4,nominalsize12,thickness.3. Notactualhardwarecertification,notmountdesigned.
- `baffled-ghost.mjs/json`:same383216raysD4/D8afterblade ->both0alternate;D8intended296832.
- `baffled-deep-ghost.mjs/json`:D8/64pupils/stride4/383104rays ->0alternate,intended315771.
- `pupil-baffled.mjs`, `baffled-pupil-8.json`:D8F165/white-raster/32pupils/5837targets zero0;ALLpartial,min6/32,p05.71875. This is raycountnotirradiance/area;needsedgepadding/overscanandrealpupilverification.
- `baffle-clearance.mjs/json`:nointersectionwithheadreinforced/fullface/bothfullbody/3mirrors;all>=.3mmthreshold. officialnearestexamined.85mm. physicalcomponentunmounted. Worldgeometrybuffers`radial-shell/baffle.*.bin`.

### Integrated into simulator (now active default)
- NEW `src/aperture-blade.js`:localgeometryfactory,worldBVHcache invalidatedbyprojectorpose/spec;finiteabsorbingblocktest.
- `src/optics.js`:visiblePath/traceRay includeapertureBladeBlocks;validateoptionalstate.apertureBlade {distance,cutDistance,size,thickness} boundedpositive.
- `src/stray-light.js`:considersbladeactualsolid asnearestabsorbing surface,kind'aperture-blade'.
- `src/projection-mask.js`:maskSignatureincludesapertureBladestate/null (alloldsignatureschanged;newcachedsignatureupdated).
- `src/app.js`:blackbladeMeshchildofprojector,updatesgeometrywithspec;automaticallyfollowsprojector.
- NEWtestaperture-blade.test.js checksfiniteclipping,centerpass,outsidefiniteextent,projectortranslation/cacheinvalidation,validation. ALL22testsPASS.
- `baffled-candidate.json` basedmarginstate addsblade. `audit-integrated-baffle.mjs/json`:coreimplementationall255542facevertices/centroidsFULLshellofficialclear,allTripo/officialouterclear;finitefield103samplechecksalsodone.
- `publish-blade.mjs`:addsbladeconfigtoassets/compact-optical-v2/layout.json ANDrecommended-layout.json. Keepssource270/displayoutside/currentheadbevel0.6. Updatesmaskcachekey toincludeactualblade. Existingdualbodymaskconservativelyretainedbecausepureabsorbercannotcreatearoute,alltargetcoveragerechecked. Manifestexperimentalcautionunchangedplusbladeinfo.
- HTTP GETrecommended-layoutverifieddesigncompact-optical-v2,fullTripo,blade present,rotation270. Latestbladebrowservisualnotyetrefreshed/inspected (previouswholefaceGPUwarpwasvisuallyverifiedlastturn). CurrentdefaultmainassetheadNOTreinforcedyet.
- PublicCOMPACT_REDESIGN_PROGRESS.mdappendbladecomparisonandlimits;rowupdatednolongerclaimsD8ghostremainsAFTERblade. READMEtopnowv2three mirrors+blade+fullwarp insteadofoldflat-onlysearch.

### Neck reinforcement prototype, NOTpublished
- `wall-normal-displacement.json`:oldbevel0.6 normal-projectedcorrespondingvertexseparationmin.425967mm,47verticesbelow.8,allnearbackneckz42.7 x~-70. Mosthead~1mm. Notglobalminthicknessmeasure.
- `radial-shell/reinforce_neck.py`:additionalinward-normalcorrectionfor75innervertices upto.574mm,outerunchanged. `bevel-reinforced.*.bin` and reinforcement.json. NEWheadnominalnormalprojectedthickness1mmneck.
- `audit-reinforced.mjs` ->reinforced-shell-audit.jsonall255542fullshellclear.
- `reinforced-assembly-clearance.mjs/json` fullreinforcedhead/fullface/allbody+mirrors>=.3mmthreshold.
- Noactualglobalminthickness/self-intersection orBlenderheadfaceintersectionQAyetforreinforced. Needfinishbeforeexportreplacingpublishedhead. Oldbevel0.6headhadexactouter+0nonmanifoldand0headfaceintersectBlender.

### Next meaningful steps
1. Checkbladeinactualbrowser/fullfacewarpstillcorrect, exposeclearstatusofexperimentalblade (currentlyonlyblacktinygeometry+docs);ifneededaddeditable/toggleUIwithconsistentmaskinvalidations.
2. Finiteilluminationedgepadding: currentwarprastertrimsboundary, D8min6/32 thoughnozerofield. Canextendwarpcolors2nativepixels usingtransparentoffscreenbackground+drawshiftsbeforemask; MUSTretestemittedghost/coverageafterpadding. CurrentFaceWarpWebGLRendereralpha:falseblackbackground ->needalphatransparentfortextureedgefill; notimplemented.
3. NeedactualnativeGPUframeverificationnotCPUapprox, finerindependentpupils/focus/exitpupilposition sensitivity. Bladeplaced1mmfrommechanicalrim,assumedexitpupilatrim. Realpupilposition/diameterunknown. Couldassesvariationsbutcannotclaimrealcalibration.
4. FinishreinforcedheadQA/export/fullassemblyBlender,actualmount/cabling/coolingconstraints. Baffleand3mirrorshavepositionsbutnopracticalmountdesigned. Blade is small~12mm triangularabsorber,notwholeframecrop.
5. Preciseeyes/nose/mouthlandmarkalignmentstilluncalibrated;fullfaceisvisiblefromlive sourceandnotpartialscreen,butlookatfrontalviewforpositions.
Do notmarkcomplete;realhardwareachievementunproven. Noexternalmessages/installations.

## USER REDIRECT — supersedes head-only miniaturization
User explicitly abandoned placing projector in rear head, asked for a Furhat-like lower torso concealing projector, then explicitly said proceed WITHOUT convex lens. Interpreted and stated: no added convex lens, REMOVE previous convex mirror too, retain projector stock lens and use FLAT MIRRORS. User accepted prior flat-mirror plan via this steering. Preserve face/head exterior and2parts,adult110%,full135x75module,wholefacevideo,goalcompactallface. Projector now permittedinlowerTORSO. STOP optimizing oldhead-onlyconvex+blade route. Goal objectivegeneralcompactstorage stillcompatible; latestuserconstraints takeprecedence over stalehead-onlynotes. Oldmainappconvexcandidate remains untilnewflatvalidated;do notpresentitasnewplan.
Reuse `work/flat-redesign/candidate-layout.json` as flat startingpoint:twoFLATmirrors,wholebodyP[-30,0,-60],rotation[45,0,180],main[-27.51,.421,212.887],130x155trimmedflat;firstfold[-29.65,-7.60,-54.24],12.3x16.45. Previouslyrejectedfor182mmneckextension,nowtorsoallowed. All255542outertargets/actualbodyclear,originalwallonly75.7%. Newworkdir `work/torso-flat` for torso+flat redesign,don'treuseconvex-tailoredheadwithoutaudit. Needinnerheadrayalignedrebuildforflatfield,torsoopaquehousingandneckaperture,fullsourcewarp/mask/finitesamplevalidation. Mainappmusteventuallyswitchflat+torsomesh,fullstockprojector,noaddedlens/nocurvedmirror.
Latesthead-onlyreinforcedQA:0outercoordinatechanges,0open/nonmanifold/degenerate;9connectedcomponents(222868main+8tiny6-8verts). Reinforcedfaceintersection0. Tinyislandinvestigationstartedsmall-components.jsonpending/nowprobablycomplete,notpublished. This is nowsecondarytoflat/torso,mayneedfixonlyifnewheadbuildinheritsoutertinyfans.

## Movable-neck study completed this turn
- New /neck-study.html + src/neck-study.js: functional yaw/pitch/roll sliders with full Tripo body, flat mirrors, head/face, representative paths and conservative swept body envelope; screenshot verified rendering.
- Main index links new study but still old convex design; don't describe old main as latest architecture.
- work/torso-flat/audit.mjs completed: 254817/255542 (99.7163%) full-shell official chief samples, 725 blocked. Need solve these, then finite-pupil/ghost/fullface warp integration.
- work/torso-flat/neck-envelope.mjs sampled 8177 poses: body-only AABB 299.38 x 307.24 x165.67, with10mm eachside319.38x327.24x185.67. No completed physical torso or neck mechanism. Pivot[-15,0,42], proposed yaw45,pitch20,roll15. Motors/rigidity/wires/ventilation/collision unverified.
- UI assets in assets/torso-flat-study, read MOVABLE_NECK_STUDY.md for caveats.
- Sessions14402 and84106 complete. No active shelljobs started this turn remain.
- CUA neckTab tab7 visible http://127.0.0.1:8769/neck-study.html, visually verified. Main tab4/oldcandidate5 remain.

## Torso-flat interior continuation (progress, not complete)
- Diagnosed725 prior failures: fullFace inner self-crossings/ripples, NOThead. blocked.json records3facecrossings each.
- Radial-face0.8 prototype passed255542 butmaxinnertravel39.94mm; rejected for adoption. Smallfractionradial.003/.005 too thin(minnormal.025/.041mm), also notadopted.
- Smooth original inner ONLY with outerverticesfixed. 20iterations clear255521;40clear255540;80clear255541;160clear255540;320clear255539. Localv1 same1fail due weighting onoriginalcoordinates.
- local-face-v2.py starts80iterations andsmooths6mmradius around[14.085,-62.874,107.804] using currentvertexcoords. face-local-v2.*.bin finalexperimentalface. audit-local-v2-full.json255542/255542plannedpaths. local-v2-qa.json exact originalouter169900tris;open0/nonmanifold0;maxinteriormove6.58257mm,92verts>2.6mm. Blendercheck-local-v2-assembly.py intersection0. Selfintersection/minphysicalthickness stillunverified.
- Publishedfacev2only inassets/torso-flat-study, auditupdated; originalSTL/oldmain untouched. Neckstudy says予定光路, notactual100% success. MOVABLE_NECK_STUDY.md appendedupdates.
- Nonseq1023chief sample:1018face5head. head-seam-sweep.json provesall5headhitpositions coincide exactly withtarget(<5e-13mm):sharedface/headboundary; changingbevel.6to1doesn'tchange. DoNOTsilentlyrelaxtracerepsilon or falselycallthesephysicalinternalblockages. Finitepupiledgesstillvignette.
- Pupil1023x64ray D8/F350:58796face/65472,head412,escape2923,mirrorback3341. D2/D4/D6detailsinfullaudit. Pupilplane/diameter/focus hypothetical.
- build-mask.mjs made native9chiefsubsamples/pixelmask usingRADIALexperimentalface; all255542plannedtargetsremain. work/torso-flat/nominal-mask.bin/json. Savedexperimental-mask.binUIassetsBUTnotintegratedintoactualdisplay.
- audit-masked-bundle-v2.mjs tests thismaskwithNEWsmoothedfacev2:stride8nativepixels16pupilsD8F350,134448rays,ghost0,face94803/head21304/escape9476/mirrorback8865. Sparseconditionalnotactualghostproof. Unknownrealexitpupil; specularfrontmirrorsassumed.
- Allshellsessions thisturn terminal incl49679,4034,68692. Nopendingjobs. No installs.
- Next:finitebundleaperture/brightness/focusandfootprint pertarget, normalizeunusedprojectorpixels usingfullfacewarp, torso+3axisphysicalmechanism/clearance optimization. Currentneckstudykinematicsonly,noactualtorso/noactuators. GoalACTIVE.

## Latest USER STEERING: torso compactness is priority
User says eyes/nose/mouth can be fixed later by projection mapping; focus on how compact lower torso can be. Explicitly deprioritize landmark mapping. Keep whole-face light and humanlike3axis neck; do not silently sacrifice coverage.
- Optional async question pending: maintain proposed yaw45/pitch20/roll15 or reducepitch/roll for compactness? Untilanswer maintainoriginalranges. Theseangleswereassistantproposed,notuserexplicitfixedspec.
- NEW livevideo integrated into neck-study.html using src/neck-live-projection.js + static face.projection-uv.bin/illuminated/outerindices/projection-metadata. Uses FaceWarp allsource no crop, /api/frame fromactualオーキャンv2,rotation270. CUA screenshotverified actualfacevideo onhead. Approx geometry preview, no radiometry/defocus. Mainrootoldconvex unchanged. CUA neckPreview tab8 nowvisible/currentuser. /api/statuslive semanticreadytrue.
- Existing displayedstudy stilloriginalflatP[-30,0,-60] smallfold12x16. Do notcallitcompactnewcandidateuntilassetsupdated.

### Aperture experiments before compactness steering
- fold-focus-sweep.json: D8ideal,F250/300/350/400,512fieldsx32pupils, margins0/1/2/4/8. Largerfoldimprovesvignetting; margin>=2collidesofficialbody,margin4/8alsoTripo. fold-margin-clearance.json.
- SimplymovingP alongfoldnormal20mm, plane10mm failed49%frustum becausefootprintcenterwaswrong! rejectedshifted-layout.json,audit-shifted.json. DoNOTpublish.
- recenter-fold.mjs fitscorrectfootprint aftersamevirtualpose-preserving shift. recentered-layout.json selectedshift10normal,rectangle25.70x35.03. Bothbodies/head/face surfacecollisionfalse; threshold.3clear. Mainmirroralso.3clear (main-mirror-clearance.json).
- audit-recentered.json COMPLETE:255542plannedpointsallclear,bothbodies. D8/F350:64929/65472correct,474head69escape,ghost0. D4:65098face374head; noescape/mirrorback. 3chiefhitsarelikelysharedseam asold5,notyetclassifiedthislayout.
- masked-recentered-audit.json oldmasktestedwithnewcandidate:134448rays,ghost0,106562face,26960head815escape111mirrorback. Conditional.
- build-recentered-mask.mjs COMPLETE session52981:9chief/nativepixelmaskall255542targetsremain0missed, filesrecentered-mask.bin/json. Thisisinworknotpublished.
- export-recentered-projection.mjs COMPLETE session40672 outputwork/torso-flat/recentered-projection,all167698vertslit. RecenteredneckenvbodyhullBOXapprox305.57x302.28x168.51beforepadding. Notpublishbecauseusernowprioritizescompactorientationsearch.

### Compact torso optimization CURRENT WORK
- optimize-torso.mjs samples4896 foldnormal/distance variants preservingvirtualprojectorpose (andmainmirror/head) ->476coarsefeasible. Bounds allbodymaxz32,minz-210; dynamicbodymaxz25. Proposedpivot[-15,0,42], yaw45pitch20roll15. torso-ranked.json top80. ThisuseslocalbodyBOXcorners+sparseopticalaperturefit, notfinalgeometryproof.
- screen-torso-ranked.mjs COMPLETEsession93110, torso-screened.json. Unique candidates physicalfold-body/head/face surfaces tested and857facepointsbothbodyclear. Manypass. Firstboxscore15.939collides; score16.1357and16.5254pass.
- projector-hull.mjs COMPLETE: exactconvexhull of ALL1,627,768 vertices ofBOTHcompletekit+Tripo meshes ->309vertices. projector-hull.json. Containsbothmeshes, no morphology approximationbeyondconvexenclosure.
- rank-hull-envelope.mjs compares top80using105poses exacthullsupport and10mmmargin eachside. torso-hull-ranked.json.
- BEST exacthullcandidate scoreoldbox16.5254; currentcompact-body-layout.json P[5.3930856,-.8843435,-43.3796753], seefileforrotation/mirror. Passed857samplephysicalchecks in torso-screened.json (nearidenticalduplicatepositionsfloatingprecision). Hullenvcoarse269.95depth265.95width193.81heightmm (13.914L).
- compact-body-dense-envelope.mjs COMPLETE:8177poses2.5deg yields13.9933L candidatevs16.9931Loriginalsamehullmethod(~17.65%decrease). filescompact-body-dense-envelope.json (array candidate+baseline). Needinspectexactdims.
- RUNNING fullauditcompact-body session18747 (audit-compact-body.mjs ->audit-compact-body.json), and maskedfiniteaudit session42608 ->masked-compact-body-audit.json. Pollsamehandles,don'trestart.
- RUNNING pivot-sweep.mjs session55325 ->pivot-sweep.json:12pivotpositions x[-35,-15,5],z[10,20,30,42] y0. Headbasez42.1 so joint0–32mmbelowbase; samefullranges8177posesbodyhull, no finishedjointmechanism. Mightreduceenvelopefurther. Consideranatomicalcenter & actualhousing/headcollisionbeforepublishing.
- Staticprojector+singlemovingfold study static-projector-screen.json:792variants524physicalboxfits,27neckposes43facepoints, bestmincoverage48.8%; REJECT asdoesn'tpreserveallface. Projectoryaw-onlycompensatingpitchroll versionyaw-only-projector-screen.json best74.4%,alsoreject. Bothinfinitefoldaperture/sparsefrustumonly so notfeasiblecandidates. Do notclaimcompact3Lsolution.
- No physicaltorso/neckactuator/mount yet. Supports/cables/ventilationstillunmodeled. Needfollowusercompactnessfocus, nofurtherlandmarkpolish.

## Torso-v3 PUBLISHED: current compactness result
- Userprioritybodycompactness remains. Optionalrangequestion pending(noanswer); maintain45/20/15.
- All jobs in priorcheckpoint FINISHED including18747,42608,55325. Allnewjobs FINISHED incl49546,84532,97632. No live shelljobs now.
- compact-body-layout earlier13.99Lbodyonly candidate was superseded after INCLUDINGsmallmirror inranking. complete-torso-ranked.json top2 selectedsecond: avoidsraisedupperpack,254x253x163coarseboxinclsmallmirror.
- CURRENT state work/torso-flat/torso-v3-layout.json; copiedtoassets/torso-flat-study/layout.json. P[-23.1077677,-17.8731346,-56.5250105],rot[-63.8835585,-35.6268416,-146.0937038]. Smallrectmirror23.8282x35.1849 center[-30.97824,-4.73006,-44.47165],r[56.30993,25.65891,16.10211]. Mainoriginalflatunchanged.
- PIVOT[5,0,10],32.1mmbelowheadbase. Allrigidhead+fullbody+mirrorsfollowit. Noactualmotors/neckmechanismyet.
- torso-v3-envelope.mjs8177poses2.5deg usesactual309-vertexhullfrombothfullmeshes+smallmirror8corners+lensorigin. continuous-envelope.mjs adds rigorousaxisguard.349702mm perface basedmaxradius163.27/Taylorsecondderivativebound; then10mmpaddingeachface. FinalWIDTH253.8503 DEPTH255.1224 HEIGHT164.0613mm (10.625L),baseline17.176Lsameconsistentmethod38.1%reduction. Allbody+smallmirror included, nohardware/cooling/outerwallthickness.
- torso-v3-head-clearance.json analyticalheadexteriorminZ5.94003 forfullcontinuousyaw/pitch/rollrange; enclosuretop3.80161,verticalgap2.13842mm. Noactuator/collar/toleranceproof.
- audit-torso-v3.json full255542chiefplannedpointsallpassbothbodies. D8F3501023fields64pupils:64831face/65472,471head170escape;3chieftiesheadlikeboundarynotreclassified. masked-torso-v3-audit.json134448oldmaskbundle:ghost0,106605face26857head884escape102mirrorback. Conditionalnotactualcoveragevalidation.
- torso-v3-uv-invariance.json all167698vertexUVmatcholdmapmax2.98022e-8 no missingpaths. ReusedpublishedUV/lit/maskwithupdatedmetadata.layout; oldmaskconditionaltestsdocumented. Nolocalmodelmass/guaranteedbrightness.
- Public neck-study.html updatedheader smalltorso,compareoldboxcheckbox. src/neck-study.js readsPIVOTfromenvelope,showshullboundsvolume38%comparison. scene/currentassets havev3geometry&state. Liveavatarstillworks withreusedfullwarp, butdon'tspendtimeonlandmarks.
- CUAcompactTorsoTab tab10 visible http://127.0.0.1:8769/neck-study.html?layout=torso-v3; AXandscreenshotverifiedcurrentdims/head/body/video. EarlierneckPreviewtab8maystilloldloadedstate; maintab4oldconvex unchanged. Avoidmoreduplicatetabsifpossible.
- Nextmeaningfulwork:designactualtorsohousingwithnecklightopening,body+mirror+support/gimbal/cable/thermalclearances. Need accountneckmovementkeepinglight; eye/nose/mouthmappingdeferredperuser. Preferthisv3overoldrecenteredcandidate orbody-onlyscore. No completephysicaltorsoyet. GoalACTIVE.


## 胴体外装と首開口の追加検証

v3を維持し、固定外装の試作STLを出力した。外寸は幅253.850×奥行255.122×高さ164.061mm、壁・底・天板3mm、天板の首開口は前後101×左右145mm。開口は全可動域の両平面鏡間の光束包絡から計算し、角度補間の座標補正1.27781mmと開口の追加余裕4mmを含む。外装メッシュは24頂点44三角形、開いた辺・非多様体辺0、体積は解析式と一致。

出力メッシュを独立に検証：105姿勢、顔343点、光路108045区間で外装による遮断0。27姿勢×2種類の全体プロジェクタモデルの計54干渉確認で交差0。連続包絡の検証を補う疎な独立検査であり、実機の公差や首支持機構を保証しない。

首支点を前方へ移す8候補も比較した。最小の疎な格子評価では9.102Lだが、頭の最低位置が水平な外装上面を下回る。これだけで実際の交差を断定しないが、現行の上下面分離条件を満たさないため未採用。現行支点の鏡間光束包絡からの距離は約9.01mmで、中央に大きな首機構を置けるとは限らない。前方移動はこの空間を広げる一方、上部外装の形状と頭の動作の再検証が必要。

試作STL：assets/torso-flat-study/DANYA-torso-envelope-prototype.stl。首開口を覆うカバー、首機構、支持材、配線、放熱、締結は未設計であり、製造用の完成形ではない。目・鼻・口の位置合わせはユーザー指定どおり後回し。

Completed sessions: 13985 housing audit; 87546 pivot sweep. Reports: work/torso-flat/housing-integration-audit.json, pivot-mechanism-tradeoff.json. No audit processes remain from these sessions. Published pivot remains [5,0,10]. Goal remains active; mechanical design unfinished.


## 継続探索：v4候補（未公開）
前ターンは外装検査と支点比較の証拠を追加したprogress。今回は支点[5,0,0]候補を連続補正付きで評価。可動範囲は維持し、外寸候補は幅245.5599×奥行247.0592×高さ163.7139mm、9.93217L（v3より6.52%減）。頭部内外壁頂点の最低高さ5.01676mmに対し外装上面2.86030mm、上下分離2.15647mm。鏡間光路を585姿勢・384セルの補間補正1.32965mmで包絡し、3mm天板の開口を103×151mmへ拡大。候補STLはwork/torso-flat/v4-housing/。開いた辺・非多様体辺0、解析体積一致。公開版は検査済みv3のまま。
首の軸配置を光学的に予備検討：支点から130mmの直線で、左右・前・下は鏡間凸包との連続距離下限10mm超、後方・上方は凸包へ侵入。これは本体やブラケット、複数軸の相対回転を未評価の候補選別で、首機構成立の証明ではない。v4-shaft-optical-screen.json参照。
v4の独立外装BVH検査はexec session62040で実行中。次回は同じhandleをpollし、結果を確認してから公開版変更を判断する。首支持・駆動・配線・放熱・実光学の不確定性が残り、goalはactive。


### v4首機構の予備配置（継続）
左右85mmのピッチ軸中心線を、内側ロール軸の±15度について0.5度刻み、中心線0.5mm刻みで検査。座標距離のLipschitz補正0.62099mmを差し引いても、鏡間凸包から左右それぞれ9.4465/10.2179mm、全体本体凸包から18.4739/39.4866mm以上離れる。径8mmの仮軸はこれら2種類の包絡には当たらない見込み。ただし鏡実体、頭壁、軸受、支持アーム、荷重は別途未検証。軸方向Xの後方は光路、Z下方は本体へ入るため、単純な中央下向き軸は不採用。前方片持ちロール＋左右ピッチ＋外周ヨー支持を次に検討する。
固定屋根の現在開口は光路用のみ。例えば左右55mmの軸でもヨー45度でx約44mmまで回り、開口のx最大21mmを越える。このため径8mm軸を含む首開口の再設計が必要で、機械軸を現在の公開モデルに無条件で追加していない。
検証ソース: v4-pitch-axis-optical.mjs/json, v4-pitch-axis-body.mjs/json, v4-shaft-body-screen.mjs/json。v4外装audit session62040はこのターンにpoll後もlive、ps PID14015 CPU93.9%経過1:39で確認。再起動せず次回同handleを確認する。公開版は依然v3。


### v4左右軸用の開口拡張
前ターンは首軸の光束・本体との距離評価を実行したprogress。今回は直径8mm長さ170mmのピッチ軸を左右に設ける仮定で首開口を155×184mmへ拡張（x[-82,73],y[-92,92]）。外寸はv4のまま。軸はRzRyのY軸でありpitchで方向が変わらず、rollは内側なのでshaftの高さ範囲はz±4mm。yaw±45度の連続包絡に3mm余裕を付けて開口を計算。work/torso-flat/v4-shaft-housing/に別候補STLを生成し、既存v4光路用外装は変更していない。
独立メッシュ検査181 yaw姿勢で軸と外装の交差0。光束凸包との補正済み距離から軸半径を引いて最低5.446mm、本体凸包から14.474mm、頭部最低zから1.0168mmの理想幾何間隔。軸受・駆動・腕・配線・荷重・実機誤差や他経路迷光は未検証。audit-pitch-shaft-housing.mjsとv4-shaft-housing-audit.json参照。公開モデルはv3のまま。
長時間v4本体外装検査session62040はpollでrunning、ps PID14015 CPU92.1%経過3:49を確認。次回同じhandleを継続確認。再起動していない。goal active。


## 現在公開：v4＋左右仮軸用開口
session62040は正常終了。housing-v4-integration-audit.jsonで105姿勢・108045区間の外装遮断0、54全体本体検査の交差0を確認。独立検査した光路用外装から屋根の材料だけを追加除去した155×184mm開口版を公開採用（新規の遮蔽物は増えない）。軸受等の成立は未証明。
公開assets/torso-flat-study/neck-envelope.jsonはpivot[5,0,0]、幅245.5599×奥行247.0592×高さ163.7139mm、9.93217L、元の配置比42.173%減。torsoバイナリとSTLをv4-shaft-housingから反映。torso-manifest.jsonに独立検査結果と材料除去による検証継承の範囲を添付。光学配置・顔映像UVは変更なし。v3以前の公開ファイルはwork/torso-flat/v3-published-backupに保存。
既存ユーザーtab9をreloadし、9.9L・42%減・支点下42mm、最新寸法と開口、オーキャンv2稼働をAXで確認。yaw45/pitch20/roll15でスクリーンショット確認後、正面に戻した。仮軸は表示していない（開口空間のみ）。検査プロセスは終了済みでpendingなし。
次の実作業：前方片持ちroll支持と左右pitch支持・外周yawを構造として接続し、軸受/モーター/支持材/配線/放熱/首カバーを含む収納検証。実機射出瞳・合焦・光束損失・迷光・薄肉シェル製造性は未解決。目鼻口の位置合わせはユーザー指定で後回し。goal未達成、active。前ターンも今回も証拠と成果物を追加したprogress。


### 首支持フレームの候補探索
前回はv4公開反映まで完了したprogress。今回はU字pitch carrierを3案（前辺x25/35/45mm）検査。候補x25: 中心線[5,-85,0]→[25,-85,0]→[25,85,0]→[5,85,0]、半径4mm。共通yaw/pitchを消去したhead座標でroll±15度、0.5度刻みと中心線1mm以下刻み、距離補正0.870983mmを適用。管外面までの連続距離下限は光束凸包24.860mm、本体凸包10.145mm、頭部壁26.019mm、顔壁39.996mm。さらに管全体のhead座標最大z25.9997mmは頭壁最小42.1055/顔壁最小47.3888mmより下で、材料内に含まれるケースを除外。yoke-screen.mjs/json参照。前案より検証対象を実際の接続部材へ拡張した。
U字carrierのyaw/pitch包絡＋管半径4＋余裕3mmを含む開口169×190mm候補を別途生成。v4-yoke-housing/のSTLは外寸を維持、open/nonmanifold0、解析体積一致。公開版はまだ左右仮軸用155×184開口。次はyokeと外装の独立干渉検査と、前側roll軸受および外側yaw/pitch支持の接続。構造荷重、鏡/プロジェクタの取付、モーター、ケーブル、冷却、首カバー、実機光学は未確定。進行中プロセスなし。goal active。


### 公開更新：U字支持候補と169×190mm開口
前ターンはU字支持材の連続距離と開口包絡を追加したprogress。今回はaudit-yoke-housing.mjsを実行、yaw37×pitch17の629姿勢、4円柱の計2516組で外装交差0。結果yoke-housing-audit.json。仮軸＋U字carrierの材質形状のみであり、軸受/締結/荷重/駆動やhead/projectorへの接続は未成立。
v4-yoke-housingの外装バイナリ/STLを公開へ反映。外寸9.93217Lは据置、開口169×190mm。既検査外装からroofを追加除去するだけである。torso-manifest.jsonへ支持材の距離・独立外装検査を統合。
neck-studyに任意の「首支持フレームの候補を表示」を追加。carrierはRz(yaw)Ry(pitch)で回り、内側optical rigはRzRyRx(roll)。配置検討用、軸受・駆動・固定部未設計を明示。既存tab9reload、checkbox選択、45/20/15姿勢のスクリーンショットを確認後正面へ戻した（候補表示on）。映像稼働確認。処理pendingなし。次はfront roll軸受と光学rigへの接続、外側pitch/yaw支え、荷重・駆動・配線・放熱・首覆いを含む成立性。goal active。


### 軸受ケースの候補評価
SKF公式608-2RSHの8×22×7mmを調査（https://www.emarketplace.in.skf.com/deep-groove-ball-bearing/608-2rsh）。購入/設置なし。外径28mmケースを仮定し、rollケースx12..38,y=z0、左右pitchケースx5,y75..95/−95..−75,z0の保守的カプセル包絡を評価。roll±15度、軸0.5mm刻み、距離補正0.664616mm。最小光束間隔roll2.945mm、最小本体間隔右pitch4.731mm。その他head/faceも正。全カプセル最高z38.588mm<頭壁最低42.106mmで材料外を確認。bearing-sleeve-screen.mjs/json。
現行左右シャフト±85mmは中心y±85の608幅7mmを最後まで貫通しないため延長必須。現行U字前辺はroll軸受部分で中断し、穴付きケースへ置換が必要。ケース包絡の成立を機構完成と誤認しない。公開モデルは軸受なしのU字候補のまま。
重量をasync質問（プロジェクタ・印刷顔/頭の各質量、不明回答可）、まだ未回答。寸法作業は継続可能。pendingプロセスなし。前回も今回もprogress。次はシャフト延長と軸受/留め具を含む接続・外装干渉。goal active。


### 公開更新：190mm左右軸＋軸受ケース必要空間
前回は軸受包絡を検証したprogress。今回は軸を±95mmへ延長、相対rollの連続補正後の中心線距離は光束最小9.40284、本体18.4303mm。径8mmを引いても正。軸受中心±85、幅7の両端±88.5を貫通するが抜け止め・段付き軸は未設計。
169×200開口候補を生成。左右ケース外包円柱半径14、軸方向y75..95/−95..−75。yawで|Y|<=sqrt(95²+14²)、X<=5+(95+14)/sqrt(2)、余裕3mmを加えて屋根を拡張。rollケースx12..38、半径14も含めた独立検査audit-bearing-housing.mjsで629姿勢4403組交差0。ケースは保守的な中実包絡、実際の穴・締結は未設計。body/head距離は前回検証済み。
公開外装バイナリ/STLをv4-bearing-housingへ更新、外寸は維持。画面の左右軸を190mmに延長、前辺中心をケースのため中断し、ケース必要空間を灰色半透明で表示。これを完成機構と表示していない。torso-manifest.jsonへ検査統合。既存tab9でreload、支持表示on、スクリーンショットでライブ顔映像・ケース表示確認。重量質問未回答、処理pendingなし。次は実際の穴付きケース/締結と外側支持接続、光学rig取付、駆動・強度・配線・冷却・首覆い。goal active。


### 外側yaw支持の探索
前回は軸延長/軸受必要空間の公開反映まで完了したprogress。今回、左右pitch軸受から下へ伸ばす外側yawフレームを探索。x5、y±100/110の垂直柱は本体凸包と交差し不採用。x5,y±115はサンプル距離4.743mmのみで連続補正付き成立を確認できず不採用。柱をx−35,y±110へ移した後方オフセット案が有望。
1度pitch×1度roll、中心線2mm以下間隔で再評価。前記柱+上下接続5線、半径4mm管、位置補正4.18503mmを差し引いても本体凸包から3.48479mm、光束凸包から51.1735mm以上離れる。outer-yaw-support-refined.json参照。yawは共通運動なので相対距離に影響しない。head/face壁の距離評価は別プロセスsession18249で実行開始、次回同handleをpoll（未完了なら再起動しない）。
yaw±45の柱包絡と3mm余裕+3mm壁を収める外装見積もりは幅254.094×奥行256.592×高さ163.714mm。outer-yaw-support-case-estimate.json。現在公開245.560×247.059×163.714mmに対し、支持柱まで含めると寸法が増える。未検証候補のため公開モデルはまだ変更なし。支持脚・yaw軸受/駆動・締結・光学rig取付・配線/冷却/首覆いは未確定。重量質問未回答。goal active。


### v5支持柱込み外装候補（未公開）
session18249正常終了。後方x−35,y±110,z0..−140のyaw支持管と接続部はhead/face表面から補正・管半径を引いて最低24.4208/54.0294mm。相対pitch/rollの連結範囲で表面を横断せず、初期全管z<=4<全head/face最小zなので材料内部に含まれる可能性も除外。
柱の連続yaw包絡に光路・軸受開口を合成し、3mm隙間・3mm壁・最小2mmの非構造屋根縁を追加したv5試作外寸は幅260×奥行259.5258×高さ163.7139mm。屋根開口196×250mm。前ターンの25.4×25.7cm見積もりから縁と整数mm開口の余白分増加。ケース形状・首覆いはさらに検討が必要。
v5-post-housing/にSTL・bin・QA。open/nonmanifold0、解析体積一致。独立検査v5-inner-housing-audit.jsonは内側支持+軸受包絡629姿勢4403組交差0、v5-outer-housing-audit.jsonは外柱181姿勢905組交差0。body/光束への距離はouter-yaw-support-refined、head/faceはouter-yaw-support-shells。既存光学収納領域を縮めず拡張したケース。
未実施：inner/outer支持部材同士の干渉（意図した接合/軸受接触と区別）、yaw土台接続、軸受締結、光学rig支持、荷重/駆動/配線/冷却/首覆い。公開はv4+軸受必要空間のまま。全process終了pendingなし。重量未回答。前回/今回ともprogress。goal active。


## 現在公開：v5支持柱込み外装＋内側U字修正
接続確認で旧U字端y±85がpitchケースy75..95の側壁を横切る問題を発見。U字をy±70へ変更し、半径4込み|y|<=74、ケース内端75との理想軸方向隙間1mmを確保。yaw/pitchで相対yは変わらない。yoke-inboard-screenで光束/本体/head/faceとの連続距離下限24.9257/10.3308/22.7739/40.0617mm。旧case支持アームは中心から伸びていたが、元の同一線分上x−9,y±93.75で切り詰め、ケース外面接続候補へ変更。切り詰めは検証済み管の部分集合。
mechanism-self-clearance.mjsでinner支持/shaft/rollケース vs outer5本管の線分間最短距離を正確に計算、pitch0.5deg刻みと連続補正。最小下限はshaft5.5855mm、U腕19.2516mm、front32.7468mm、rollcase77.8677mm。軸受とshaftの意図した接触・固定部の重なりは別扱いで、締結や軸受穴/肩/抜け止め未設計。
公開外装をv5-post-housingへ変更。寸法260(Y)×259.5258(X)×163.7139(Z)mm、開口196×250。manifestへ各検証追加、mechanism-outer.jsonを追加。表示の枠/外装寸法はcaseSpecに基づき、本体光学収納9.9Lとは明確に分けた。outerCarrierはyawのみ、innerCarrierはyaw+pitch、opticsはyaw+pitch+roll。既存tab9で最大45/20/15のスクショ確認後正面へ戻し支持表示on。ライブ映像あり。
全job終了pendingなし。今回も前回もprogress。次は底部yaw支持/旋回軸と光学rigをroll軸へつなぐ支持、実軸受ケース/締結・荷重/駆動・配線/冷却・首覆い。全顔照射の実機射出瞳/合焦/透過/迷光不確定性も残る。重量質問未回答。goal active。


### 底部yaw接続の探索
前回はv5公開と機構同士の隙間修正を実施したprogress。今回は後方の下桟x−35,y0,z−140から中央軸x5,y0へ回り込む3線分（下降、水平、軸）の候補を検査。baseZ−155/−160/−165、直径8mmで相対pitch/roll1度、中心線2mm以下、連続補正を評価。最小base−155で本体との管外面距離下限16.638mm、光束62.232mm。yaw-base-link-screen.json。部材長さ26mmの下向き軸までを含む。
さらに浅いbaseZ−140/−145をsession62464で検査中。次回同handleをpoll。支持経路の成立と、底部軸受ケース（仮外径28mm・長さ22mm、上下に608相当を配置）の収納/荷重成立は別。水平材の下面と固定ケース上面の擦れを避ける隙間も必要で、ケース上面を単純にbase−4にする案は採用しない。軸受・抜け止め・土台フランジを含む高さは未確定。
公開は変更なし、v5外寸260×259.526×163.714。重量質問未回答。goal active。

追記：62464は終了。base−145は管外面の連続距離下限8.52084mm（本体）を得た。base−140では下降線分が長さ0となるサンプリングの0除算を発見したため、その初回結果は無効。n>=1へ修正しsession63734で再実行中（これが次回poll対象）。NaN由来の初回ゼロ距離を物理的干渉と解釈しない。


### yaw軸受ポケット試作と有限ケース検証
63734正常終了。ゼロ長さ線分の0除算修正後、baseZ−140でも管外面の本体距離下限5.06905mm、光束59.08025mmで通過。−145は8.52084mm。初回NaNに基づくゼロ距離は無効で置換済み。両案とも棒材の経路のみでケースの太さは別。
yaw-case-body-screen.mjsをsession40559で実行中。base−140/−145、軸受ケースを半径14高さ22の有限円柱、上面base−6/下面base−28としてbody凸包との最近接三角形距離を1271 pitch/roll姿勢で評価。最近格子回転と円柱多角形誤差の補正を含む。次回は40559をpoll、再起動しない。
work/torso-flat/yaw-bearing-pocket/bearing-pocket-dimension-prototype.stlを新規生成。外径28高さ22、両端22.2径×7.2深さ、中央貫通8.6径。608規格の寸法を基準としたクリアランス仮定の試作。768頂点1536三角形、非多様体辺0、解析多角形体積と一致。実際のはめあい・抜け止め・分割締結・負荷/印刷補正未確定、完成軸受ブロックではない。build-bearing-pocket-prototype.mjs、同manifest.json参照。
公開v5は変更なし。前回/今回ともprogress。重量質問未回答。goal active。


### v6底部ケース収納候補（未公開）
40559は正常終了。有限円柱ケースr14,H22、base−140の中心[5,0,−157]でbody凸包とのサンプル最短9.99482mm、連続pitch/roll+64角形補正2.95918mmを引き7.03564mm。base−145は11.1340mm。yaw-case-body-screen.json。前ターンも今回もprogress。
予定光路の鏡前面2枚と仮8mm瞳正方形を含む包絡を連続姿勢で評価し、最低z−73.2183mm、ケース上面−146との垂直隙間72.7817mm。yaw-case-optical-height.json。別経路迷光/散乱・実瞳位置は含まない。
v6-yaw-base-housing/を新規出力。外寸幅260×奥行259.5258×高さ176.8603mm、床底−174/床内面−171、軸受ケース下面−168/上面−146、軸先端候補−170。既存棒材検証は−166までなので末端4mm追加部分は別途確認すること（body最低z−150.854より下なので垂直包絡で検証可能）。水平リンク下面−144、固定ケース上面−146の理想隙間2mm。ケースの土台締結/抜け止め/駆動は未確定。STLはopen/nonmanifold0、解析体積一致。
公開v5は未変更。次は末端・ケース/リンクの外装検証と土台への接続、反映判断。全process終了pendingなし、重量質問未回答。goal active。


## 現在公開：v6底部ケース込み
v6-base-housing-audit.jsonでyaw181姿勢×link/shaft/case=543組、外装との交差0。軸先端−170と床内面−171の隙間1mm、ケース下面−168と床内面の隙間3mm。追加軸末端−166..−170は全body最低z−150.8536より下で15.146mm以上のZ分離。case top−146とlink下面−144は2mmの理想隙間。固定・抜け止め・フランジや荷重は未検証。
公開外装をv6-yaw-base-housingへ更新、幅260×奥行259.5258×高さ176.8603mm。外側frameへ水平link[-35,0,−140]→[5,0,−140]と軸[5,0,−140]→[5,0,−170]を追加。固定yawCaseはscene直属の灰色半透明必要空間（r14,H22,中心[5,0,−157]）。manifestへ底部body/光束/外装検査を統合。公開は完成機構でなく必要空間の試作表示。
既存tab9をreloadして幅260奥行260高さ177表示とオーキャンv2稼働を確認、支持checkbox onでスクリーンショット確認。全job終了pendingなし。前回/今回ともprogress。次は光学rig（head+mirrors+projector）を前側roll軸へつなぐ支持/固定、底部固定・軸受肩/保持・駆動/強度・配線/放熱/首カバー。実瞳/合焦/光量/迷光の物理検証も未完。重量質問未回答。goal active。


### 光学rigをroll軸へ接続する経路調査
前回は底部v6公開反映を完了したprogress。今回はhead/鏡/projectorを実際にroll軸へつなぐ支持に着手。首近傍のhead/face頂点範囲を取得。頭奥へ支持材を伸ばす場合は、折返し鏡間だけでなくmain→faceの光を含める必要があるため、全original外面face頂点+full main mirror frontから反射後光束の凸包を生成。reflected-beam-hull.json（入力86616、凸包15431頂点30858面、minZ47.38885）。全鏡面の未使用領域も含む保守的包絡。
head-entry-support-screen.mjsで仮rod半径3、始点[45,0,0]→[45,0,20]→[entryX,0,20]→[entryX,0,100]を5案検査。entryX0は折返し光束（x0.5,z20）、10は折返し光束（z69）、20は反射後光束（z60.5）、25はhead壁（z49.5）、30はhead壁（z39.5）で隙間不足。これらを採用しない。首壁への固定接点が未定で100mmまで伸ばした探索なので、頭内部への全支持方式の不可能性ではない。次はbeamへ入る前の首元に接点を求め、頭固定と鏡/projector固定を分けて接続する。
公開v6は未変更。全処理終了pendingなし。鏡→face包絡を以前の首/底部検査へ自動適用したと主張しない（以前の部材は低い位置にあるが必要に応じ追加検証）。重量未回答。goal active。


### 首下端でのhead固定候補
前回は頭奥への5直線案を棄却したprogress。今回neck-contact-candidates.mjsでhead-wall表面の候補を探索。採用検討点は[16.9482,−35.0687,42.6266]と[17.2152,35.5275,42.5726]。反射後包絡から21mm程度離れる。元head頂点/外面を変更していない。
neck-fork-screen.mjsでX rollshaft[12,0,0]→[45,0,0]（r4）、[45,0,0]→[45,0,20]と左右接点の6mm下までの分岐（r3）を検査。中心線0.5mm刻み+.2501guard後、管外面からfold-main光束13.3599mm、main-face光束18.3243mm、本体35.6456mm、head壁2.47221mm、face壁18.3243mm。全rigが一緒に回るのでこれら相対距離は姿勢不変。実瞳/散乱/未計画迷光を保証しない。
先端は接点の6mm下で止めており、パッド/クランプ未設計。headを実際に支持したとは扱わない。次はforkと既存首機構/外装の相対roll/pitch干渉、接触パッド、projectorとmirrorの取付。公開v6は未変更。全処理終了pendingなし。重量質問未回答。goal active。


### neck forkの機構相対干渉検査
前回はforkと両光束/head/bodyの相対距離を確認したprogress。今回はneck-fork-outer-clearance.mjsを実行。fork vs公開outer7線分（底部含む）、pitch/roll1度+連続補正で管外面最小52.0511mm。forkの全姿勢包絡球radius55.4306、中心[5,0,0]は既存開口XY内・床より上なので外装を通過できる。
neck-fork-inner-clearance.mjsは有限円柱同士の最近接面距離でinner部材とのrollを評価。初回は保守的に切り欠き前のfront barを使ったため、X rollshaftとbarの交差を検出（実表示はcase用にy±14で切り欠いている）。実表示と同じsplit geometryへ修正しsession57899で再計算中。これが次回poll対象、再起動しない。初回のbar干渉は現行設計の欠陥と断定しない。自分自身のroll軸受ケースとshaftの意図した内包は除外。他部材との距離は検査対象。
公開v6は未変更、fork/padsはまだ表示していない。固定パッド未設計。goal active。


### 公開更新：neck fork候補
57899正常終了。実際のsplit U字に合わせたinner cylinder検査で、X shaft対pitch shaftの最小下限2.96148mm、fork上昇管対roll case3.83087mm、左右branch対roll case5.568/5.587mm。自分の軸受に入るshaftだけは意図した接続として除外。端点/軸方向位置が各他円柱の外部であることと正距離により全内包を除外できる。負荷/軸受適合は未確定。
neck-pad-surface-mapで首下端を12×12mm範囲・0.5mm格子で下から測定。左右の下向き法線セル面積推定55.25/56.75mm²、近傍有効投影面積83/81.5mm²。高さ約42.25..47.85で非平面。4角サンプルの推定で厳密接触面/耐荷重の証明ではない。次は形状適合padと保持方法。
公開assets/neck-fork.jsonとrig配下の表示を追加。首支持checkboxでforkも表示、先端は頭の下で止まりpad未設計と明記。外装寸法v6維持、manifestへfork検証を追加。既存tab9reload、ライブ顔映像・fork表示スクショ確認。全process終了pendingなし。前回/今回ともprogress。次は首pad/保持、projector/mirrorのrig固定、駆動/強度/冷却/首覆い・実機光学。重量質問未回答。goal active。


### 首パッド形状試作（未公開）
build-neck-pads.mjsで下向きsurface-mapセルの最大連結領域から左右padを生成。底面はfork tip Z、上面はhead underside0.5mm格子の補間−0.2mm。左右投影面積55.25/56.75mm²、各連結1成分、辺次数異常0。work/torso-flat/neck-pads/pad-0.stl / pad-1.stl、bin、manifest。
audit-neck-padsの独立三角形距離/交差検査でhead/face/body/両光束との交差0。head最小隙間左0.112976/右0.067016mm、反射後光束16.3929/16.0919mm以上。実造形誤差/接触圧/接着/機械保持は未確定。公称0.2mmは垂直方向の設計値で最短面距離と同じではない。
audit-pad-connectionsでforkとの意図した重なりとbeam/body凸包の非内包を検査（session27231）。頭/顔の非内包はpad底面が両shell最小Zより下で、連結padと対象表面が交差しないことから判断可能。次はpadの機構相対動作/外装、保持方法・造形適合の検討。公開v6+forkは未変更。前回/今回ともprogress、goal active。

27231正常終了：左padはfork branch2のみ、右padはbranch3のみと意図した重なりあり。光束/body凸包外の頂点確認true。全process終了pendingなし。


### 公開更新：首pad試作
neck-pad-motion-audit.mjsでpad全三角形を包む球（半径8.4974/8.6764）を使用し、inner/outerフレームの円柱カプセルと相対pitch/roll全範囲を補正付き検証。inner最小下限17.0709/16.7801mm、outer47.5882/47.0056mm。支点中心の全姿勢包絡半径62.8718/63.3597mmは既存開口内・床より上で、外装を通過。forkとの意図した接合は別検証済み。
公開assetsにneck-pad-0/1のbin/STLを追加、headFork groupに金色padを追加しheadと同じrig変換を使用。manifestへpad形状/fit/connection/motion監査を統合。表示は実物保持・はめあい未確認と明記。既存tab9reload、支持表示on、pad表示スクショとライブ映像稼働確認。外寸v6維持。
前回/今回ともprogress。全process終了pendingなし。次はpad保持/締結とprojector・mirrorをroll rigへ固定する経路（まだ表示上は浮いている）。駆動/重量/強度/配線/冷却/首カバー・実機光学も未解決。重量質問未回答。goal active。


### projector固定の公式CAD調査
前回は首pad公開まで実施したprogress。今回はprojectorをroll rigへ固定する検討へ進む。Tripo calibration.jsonを再読：135×75は底板のみ。AI全体形状は近似でレンズ基準や穴は厳密CADではない。公式底板STEPを発見：work/internal-projector/ti-cad/DLPDLCR3010EVM-G2_Mechanical/DLP002_DLPDLCR3010EVM-G2-ENGINE-BASE-SS.STEP。assembly treeで底板bounds[0,0,0]..[135,3.97,75]を確認。
extract-base-features.pyで単体STEPの円柱面を抽出（base-cylinder-features.json）。r3の半円候補4面、r1.2/r2.2の段付き候補等があるが、円柱面はフィレットも含み、ねじ規格や空き固定穴と断定しない。単体座標は組立座標と異なる。
既存OCP環境work/internal-projector/cad-venv/bin/pythonが利用可能と確認（インストールなし）。extract-assembly-base.pyをsession3191で実行中。公式REVD assemblyから該当baseをloc変換付きで読み、円柱の角度範囲/face orientationを抽出してbase-assembly-features.jsonへ出力予定。次回同handleをpoll、勝手に再起動しない。全体本体と基準合わせして固定経路を検討するための情報。
公開v6+neck padsは未変更。重量未回答。goal active。


### projector底板の固定候補の解釈と登録
3191正常終了。base-assembly-features.jsonに公式REVD組立座標の円柱面/範囲/向きを保存。単体→組立変換は(X,Y,Z)→(−Z,Y,X)を数値確認。半径1.2/2.2で各2πの段付き穴はCAD[13.742,0,48.0002]と[61.742,0,48.0002]、穴間48mm。assembly treeで両方に92005A023_METRIC PAN HEAD PHILLIPS MACHINE SCREWが既に配置されているため空き固定穴として扱わない。半径3の4円柱は各π/2、深さ約1.6606mmで浅いくぼみの角。単純な貫通長穴という解釈は撤回。
全664532 kit頂点のCAD/local差を確認し、lens datum[111.62,13.97,81.335833]の平行移動に8e-6mm以内で一致。projector-base-anchors.mjsで公式底面4隅と6mm下面側の支持候補6点を現在の投影配置へ変換。projector-base-anchors.json。これは穴や荷重保証のある位置ではなく受け台設計の候補。
次は裏側受け台からroll shaftへつなぐ経路、クランプ/ねじの扱い・基板/ファン/光路との干渉。既存ねじを取り外して流用する案は未採用。公開v6+fork/padsは未変更。全process終了pendingなし。前回/今回ともprogress、重量未回答、goal active。


### projector底板支持経路の候補探索
前回は公式baseとoccupied fastenersを確認したprogress。今回はprojector-support-route-screen.mjsで始点[45,0,0]（roll軸前端）から6候補底面支持中心へ半径3mm棒材を1折れで接続する探索。中継grid x−80..120,y−100..100を20mm刻み、z−20/−60/−100/−140。全本体凸包とfold-main光束に中心線2mm以下+1.0001mmguardで通過する候補を各72〜81件得た。最短152.197〜208.780mm。reflected/headは全経路z<=0より上。
最初のprojector→fold光束は別検証とし、projector-support-first-leg-audit.mjsで各anchor上位5、計30経路を仮8mm機械レンズ面瞳→full fold mirror包絡に対して検査。全件正の距離下限（最小3.5371mm）。実射出瞳の位置/寸法の不確定性をなくすものではない。
anchor1最短例:[45,0,0]→[20,60,−100]→[42.5336,37.3412,−107.9417]。anchor3:[45,0,0]→[0,40,−100]→[8.8032,3.3588,−93.5382]。全経路は首フレームとの相対roll/pitch、動作外装、受け台・保持・荷重未検証。現在の公開模型へは追加していない。次はこの30候補の機構/収納評価から同時使用できる支持構成を選ぶ。全process終了pendingなし。重量未回答、goal active。


### 支持経路の首動作・収納検証
projector-route-motion.mjs/jsonで30経路を有限円柱の首フレーム・軸受体積に対して評価。空間2mm以下と角度区間の距離guardを減算。anchor1 rank2の1本は内側距離下限1.302896mm、外側7.894528mm、側面・底面収納を通過。経路[45,0,0]→[40,80,-100]→[42.533585,37.341222,-107.941665]、棒半径3mm、長さ171.625680mm。負の下限の他候補は不成立の証明とは限らず追加検証を要する。
projector-support-roof-audit.mjs/jsonも正常終了。5度角度セルの端点凸包に回転補間誤差と棒半径を加えた上面スラブ断面はX14.5586..58.5735,Y−33.2760..51.2198mm。現在の開口X−109..87,Y−125..125mm以内、passes=true。projector-support-selected-route.jsonに結果参照を追記。
現在の胴体外寸は幅260×奥行259.526×高さ176.860mmで変更なし。支持アーム1本の候補が収まることを検証した段階であり、受け台・脱落防止・ミラー固定・駆動・冷却・強度を完成したものではない。公開模型には追加していない。次は底板下面で複数支持点をつなぐ受け台と保持方法を検討。全process終了pendingなし、progress、goal active。顔ランドマークの補正より胴体の小型化を優先。


### 底面受け台の骨組み候補
前回turnは支持アームの上面開口検証を完了したprogress。今回はprojector-cradle-rails.jsonの7辺（底面6mm下面、半径3mm）をprojector-cradle-motion.mjs、projector-cradle-optical.mjs、projector-cradle-first-leg.mjsで検証。全辺はfull body凸包とfold-main包絡から距離下限2.736mm以上、仮D8のfirst-leg包絡から18.845mm以上。5辺2→4、1→3、3→5、4→5、2→3は首機構連続角度guardと外装側面/底面を通過。最小outer距離下限3.59966mm。これらはanchor1支持アームへ連結する骨組みとなる。0→1と0→2はouter下限負で採用せず（実交差確定ではない）。projector-cradle-selected.json保存。底板との接触pad、保持、強度、冷却は未検証、公開変更なし。全process終了、goal active、次は接触部と保持を詰める。


### 受け台から底板への接触支柱
前回は骨組み5辺選定のprogress。今回はanchor1..5に半径3mm・高さ6mmの底面法線方向支柱を定義（projector-contact-posts.json）。projector-contact-motion.jsonで全5本の内外首機構guard距離が正、外装側面/底面内。projector-contact-optical.jsonとfirst-leg.jsonでfold-main最小20.1263mm、仮D8 first-leg最小15.63265mmの距離下限。機械レンズ面仮瞳に依存。
projector-contact-footprint.pyは公式REVD assemblyから底板を組立座標に変換。CAD(X,Z)=(15,65),(65,10),(65,65),(120,10),(120,65)でY=0接触円盤R3、厚0.05mmの底板内層についてCAD差体積0、各128点も内側。Y−6..0支柱との底板共通体積0、全boolean正常。projector-contact-footprint.json。底板以外の組立部品・接触剛性・実物の公差・保持はまだ未検証。接触するだけでは首傾斜中に脱落を防げないため完成固定とは扱わない。公開変更なし。次は全CAD組立との支柱/骨組み干渉、および側面保持の検討。全process終了、goal active、progress。


### 接触支柱の全公式組立検証・受け台候補の公開
前回turnは接触支柱の底板CAD/首動作/光束検証を完了したprogress。projector-contact-assembly.pyで公式REVD STEPのfree root組立shapeと5本のCAD R3,H6支柱の共通体積を計算。全5本booleanDone=true、intersectionMM3=0。projector-contact-assembly.json。底板接触は意図したもの。保持・強度・実物公差の証明ではない。session74481正常終了。
公開assets/torso-flat-study/projector-cradle.jsonに選定支持アーム2辺、下面骨組み5辺、接触支柱5本を保存。neck-study.jsに紫色のprojectorCradleをrig子として追加し、機構checkboxに連動。HTMLに候補・未保持/強度未確認を明記。公式assembly支柱結果も同assetsへコピー。node --check通過。既存browser2 tab9 reload後、候補を表示し、0/0/0と45/20/15で描画・rig追従を視認。オーキャンv2正常。最後にreset、機構表示true、markDeliverable。
胴体外寸変更なし。全顔のchief率255542/255542は従前条件の値で、受け台について別途保守光束距離を検査済みだが実機100%とはしない。次は保持構造、鏡支持、駆動/冷却等。全process終了、goal active、progress。


### 底板上側の爪・コの字保持具の探索
前回turnは接触支柱の全組立干渉と受け台公開のprogress。今回は公式REVD assemblyでtop-lip-screen.pyを実行。CADX65のZ0/70上爪6×3×5（Y4.07..7.07）は共通体積0。X120は29.60198/3.85302mm3の交差、採用不可。光路/動作が通ってもCAD部品干渉で落ちる例。projector-top-lip-screen.json完了。
X65に限定しC型保持具を上下/側面3boxで定義（projector-edge-clamp-screen.py/json）。Z0側全形状は全公式組立との共通体積0。Z70側は2.6517697mm3の交差で不採用。各パーツの個別識別は次。両側の保持具が完成したとは扱わない。中央の対向クリップだけでは長手方向滑りも未解決。
projector-edge-clamp-lines.mjs/json、motion.mjs/jsonでboxを包むcapsuleを評価。motionの半径は底sqrt18、側/上sqrt11.25。内roll1°、外pitch/roll1°nearestguardで全正、外最小3.05376mm。外装側/底内。光束は全boxをR4.25で保守評価、fold-main最小30.30105mm、仮D8 first-leg最小29.55096mm。これら合格はZ70側の実CAD交差を覆さない。公開表示は以前の受け台のまま（保持具未追加）。17342/22118正常終了、pendingなし、goal active、progress。次はZ70側3boxを個別に調べ、干渉箇所回避と長手方向止め/組付け方法を詰める。


### 保持具干渉の部位特定・v2
前回はC保持具screenのprogress。projector-edge-clamp-interference.pyでZ70側3boxを個別に検証。下面0、側面44.9487856、上面21.4583925mm3。以前の重複box compound共通体積2.65177は定量値として不適切、個別結果を優先する。候補不採用の判断は維持。
v2は下面[62,-9,62]+[6,6,17.8]、側面[62,-9,76.8]+[6,15.57,3]、上面[62,4.07,70]+[6,2.5,9.8]。側面を1.7mm外へ、上面厚さ3→2.5mm。projector-edge-clamp-v2-interference.json完了(session26629終了)：下面0、側面0、上面16.186071mm3、干渉bounds[63.953988,4.07,75]..[68,6.57,76.6002]。v2も不採用。上の跨ぎ部に逃げが必要。
重複assembly compound処理の定量不確かさを避けるため、各transform済leafと個別boxのbooleanを行うprojector-edge-clamp-v2-leaves.pyを追加。session58044が現在実行中、直近pollはRead RetDone、Transfer待ち。次回同sessionをpollし、projector-edge-clamp-v2-leaves.jsonを読む。再起動しない。候補boxとleafのCAD boundsで保守broadphase。全root探索。公開は受け台のみのまま。首動作/光路検証は旧保持具のものでv2へ流用しない。goal active、progress。


### 個別leaf干渉結果と保持具v3〜v5
前回progress＋live58044をpollして正常終了確認。v2上爪はMAIN-BOARD-MOUNT単体と16.186071mm3干渉。下面/側面は各leafの保守bounds候補なし。
v3はX57..63へ5mm移動、下面X57..68で既存anchor3に接続。projector-edge-clamp-v3-leaves.json：上爪がSHANE-PRT0016/17/19と3.17964/2.90131/3.17964mm3干渉。MAIN-BOARD-MOUNTは0。v3不採用。motion/opticsは通過したが実CAD干渉を覆さない。
CADtreeのY4.07..6.57,Z70..79.8の部品boundsを調査。X13.223..61.224には連続したSHANE部品、X74以上はengine等。X8..13候補をv4として下面[8,-9,62]+[10,6,17.8]、側面[8,-9,76.8]+[5,15.57,3]、上面[8,4.07,70]+[5,2.5,9.8]。v4-leavesでは上面MAIN-BOARD-MOUNT11.969448mm3。v4不採用。
v5はこの支持部品をまたぐアーチ：下面v4同一、外柱[8,-9,76.8]+[5,39.3,3]、上橋[8,27.8,70]+[5,2.5,9.8]、内爪[8,4.07,70]+[5,26.23,4.7]。projector-edge-clamp-v5-blocks.json等。leaf干渉：前三部品はbounds候補なし、内爪がSHANE-PRT0004と23.571243、MAIN-BOARD-MOUNTと37.6047mm3、v5も不採用。首動作全正/収納内、fold-main最小81.5107、仮D8 first-leg最小80.7606mmでもCAD不成立。
scriptセットは各v3/v4/v5のlines,motion,optical,first-leg,leaves。注意：コピーされたscope文章の固定半径表記は古い箇所あり、実装は各box最長軸のcapsule半径radiusMMを使用、motion外1°/inner1°guard。脚部は三角体積等にせずbox保守。全session58044,61018,75476,20569正常終了pendingなし。公開は受け台のみ、保持具未追加。goal active、progress。次はMAIN-BOARD-MOUNTの断面/占有領域を見て上爪の通過可能性を直接調べるか、既存の底板段付き穴と既存ねじを使う固定案を具体的CAD寸法で再検討する。現在のねじは空き穴として扱わない。胴体変更なし。


### 既存底板ねじを共締めする案へ
前回turnは保持具v3〜v5の不成立を確認したprogress。今回CADtreeとbase-assembly-featuresを再読、底板の48mm間隔段付き穴(CADX13.742/61.742,Z48.0002)に実ねじinstance18/19が占有することを確認。projector-existing-screw-features.py/jsonで当該2ねじの変換済円柱を抽出。半径1の外径、0.4mm周期のねじ形状、先端Y8、座面Y2の首下6mmに整合。web92005A023検索はメーカーの正確な製品仕様を確定できず、購入型番確定には使っていない。
下面固定ボス案：R4,H10,Y−10..0、貫通R1.2、下面座ぐりR2.2深2（座面Y−8）。既存ねじ首下6から+10=16mmの交換候補で先端Y8を維持する寸法案。現在のねじを空き穴扱いせず、交換を前提とした候補。threads/締結強度/工具アクセス未検証、実部品/公開CADのねじは変更なし。projector-screw-boss-leaves.py/jsonで各boss対個別leaf:底板/MAIN-BOARD-MOUNT体積0、その他保守bounds除外。
projector-screw-boss-lines.mjs/jsonでboss外包R4円柱と、Y−6の穴中心からanchor1/3(X15/65,Z65)へのR3接続棒を定義。motion.mjs/jsonはouterpitch/roll0.25°guardへ細分化、inner1°。全正、outer最小1.444192mm、側/底収納可。初回1°下限−0.202は未証明であって実交差とはしていない。optical/first-leg検証全正、fold-main最小39.96887mm、仮D8 first-leg39.21878mm。接続棒のfullbody凸包クリアランス、接続箇所・工具アクセス、座面の実接触・thread適合・強度は次。公開変更なし。20767/76210/43074正常終了、pendingなし、goal active、progress。


### 一体受け台CAD生成・ねじ通路・表示更新
前回はねじboss案のCAD/動作/光路検証progress。projector-screw-connector-clearance.mjs/jsonでR3接続棒2本のfull body凸包距離下限2.73778/2.73637mmを確認。
projector-cradle-cad-lines.mjs/jsonで公開旧骨組みと5接触支柱、2接続棒を公式CAD座標に逆変換。build-screw-cradle.pyで全円柱とR4boss2個を融合し、ねじ貫通R1.2と下面座ぐりR2.2深2を最後に切除（接続棒が穴を塞がないようにする）。work/torso-flat/screw-cradle/cradle.step/stl,audit.json。valid=true、solidCount1、volume13943.9582mm3。CAD座標Y−60..−8.1のR2工具軸と受け台の共通体積は両方0（取り外した受け台のみ、実工具/組立全体/ねじ頭スロット適合は未保証）。16139正常終了。
公開assets/torso-flat-study/projector-screw-cradle.stl/.step/-audit.json追加。projector-cradle.jsonのcadMesh/cadDatumで表示指定。neck-study.jsはSTLLoaderでCAD原点からdatumを引き、projector位置/回転を適用した紫色meshをrigへ追加。旧円柱描画は置換。ねじ交換・適合未確認をUI注記。JS構文チェック通過。browser2 tab9 reload、機構表示、0/0/0と−45/−20/−15で描画と追従確認、ライブ映像正常。最後reset・機構true・markDeliverable。
まだ元のprojector Tripo本体に見える既存ねじ形状は交換表現にしていない。交換ねじの具体品/ねじ込み先/締結荷重/座面/工具アクセス（装着状態）とroll shaft端への保持は未検証。ミラー固定、駆動、冷却も残る。胴体外寸不変、実機全顔100%未証明。全process終了pendingなし、goal active、progress。


### 交換ねじメーカー寸法と受け側
前回turnは一体受け台生成・公開のprogress。projector-thread-receiver-features.py/jsonでMAIN-BOARD-MOUNTとFAN-MOUNTのねじ軸近傍円柱を抽出。相手はFAN-MOUNT-SS、各軸R0.8、CADY3.9738..5.9738の深2mm（単純円筒、雌ねじ形状は未再現）。M2ピッチ0.4なら名目5ピッチ区間だが実噛合い/強度ではない。46016正常終了。
Aspen公式図面https://www.aspenfasteners.com/content/2D_PDF/product61/ME031-204X16.PDFをwebで確認。ME031-204X16はM2−0.4×16、全ねじ、A2、Phillips H #1、頭径最大4mm、頭高1.60±0.12mm。座ぐりD4.4深2は名目半径隙間0.2、最大頭高余裕0.28mm。新座面Y−8+首下16で旧先端Y8と名目一致。長さ公差・実雌ねじ・材質/引抜強度は未確認。元はファン支持部の締結で、ロボット動荷重適合は保証できない。projector-fastener-fit.jsonをwork/publicassetsに保存。購入/取り付けは行っていない。表示形状変更なし。次は座面荷重経路・締結への動荷重を整理し、ミラー支持とroll軸固定等の未設計箇所を詰める。全process終了pendingなし、goal active、progress。


### 小鏡支持アーム候補
前回turnはメーカーねじ図面/相手穴の確認progress。今回は未支持のfold mirrorへ進む。fold-mirror-support-screen.mjs/jsonは受け台肘[40,80,−100]から小鏡背面中心−8*normalまでR3棒を探索、fullbody凸包とfold-main包絡に空間2mmguard。224経路、直線119.980234mmが最短。fold-mirror-support-first-leg.mjs/jsonで仮D8 firstleg下限3.9999007mm。motion.mjs/jsonは首機構continuousguard正（inner44.84568、outer23.50271mm）、側/底収納可。selected.json保存。鏡厚み・接触/保持/調整・部材剛性未検証、公開未追加。
探索中の始点=waypointで長さ0辺が生じ、コピーしたfirstleg/motionのn=ceil(length/2)が0でNaNになったため、今回2scriptをmax(1,ceil(...))へ修正して再実行、全5候補有限値。directrank0の結果は変化なし。重複経路rank1は同一解で独立根拠にはしない。旧検証のnonzero経路に影響なし。72693正常終了、全process終了pendingなし、goal active、progress。次は鏡厚み/背面受け/固定方法を加えた候補を検証して可視化する。


### 小鏡厚さ・裏面受けの検証と表示
前回は直線小鏡支持候補探索progress。今回は小鏡の仮厚2mm（前面反射面0/裏面−2）、接着余裕0.2mm、裏面disk R5 Z−4..−2.2、stem R3 Z−8..−4を定義。fold-mirror-backing-lines/body/motionおよびfold-mirror-thick-lines/motionを実行。fullbody凸包から受け距離下限14.6052mm以上。内/外首機構capsuleguard正、厚み入り鏡も内28.3412/外58.0586mm。鏡capsuleを含め動作外装側/底内、maxZ−5.5374以下で屋根内面−0.1397より下。
fold-mirror-thickness-audit.mjs/json：全2mm厚小鏡箱とfullbody凸包surface距離3.333934mm、鏡頂点のbody内包false（面交差/鏡内へのbody全包含はbox小サイズとbody全体extentで除外、厳密絶対公差証明ではない）。仮8mm瞳とmainmirror全頂点の小鏡表面normal半空間距離最小3.00942/52.82367mm。対応するconvex光束は表側、disk/stem全体は裏側にあり予定first/fold-mainを遮らない。異常迷光/実射出瞳は未保証。
公開fold-mirror-support.jsonに支持直線とdisk/stem、厚2を保存。neck-study.jsのprojectorCradleに紫色支持円柱・裏面箱追加、機構toggleに連動。HTMLに仮厚2、接着/調整/保持未検証を明示。syntax通過、browser2 tab9 reload/AX/スクリーン確認、live映像正常。最後0/0/0、機構true、markDeliverable。cradle.step/stlは小鏡支持追加前の受け台のまま、混同しない。元鏡の反射前面/位置は不変。全process終了pendingなし、goal active、progress。次は小鏡受けの剛性/調整/保持、主鏡の支持、軸固定等。


### 主鏡の厚み検証開始
前回turnは小鏡厚み・受けの検証と公開を完了したprogress。今回main-mirror-thickness.mjsを追加し、974点trimmed輪郭、反射前面固定で裏側へ0.2/1/2/3mm押出し、head/face内壁meshとの最近接面距離を比較。寸法129.8459×155.4032mm。surface距離のみでmaterial内包は別検証を要する。
実行session91754が現在live、直近write_stdin pollはsession返却・未出力。未完了/エラーではない。次回同sessionをpollしmain-mirror-thickness.jsonを読む。再起動しない。mirror querygeometryにboundsTree未設定のため低速の可能性があるが、計算自体の結果は無効ではない。将来の別実験で高速化するならgeometryへboundsTree追加を検討。公開変更なし。goal active、現在specific live process待ち。


### 主鏡計算高速化・裏面支持点探索
前回turnは主鏡厚み計算開始とlive91754待ち。今turnでsamehandleがliveと確認、psでPID38735(経過3:10 CPU73.9%)を観測。実装のquery mirror geometryにboundsTreeがなくclosestPointToGeometryの低速fallbackが使われるとライブラリsourceで確認。性能上の原因を修正するため意図的にSIGTERM、91754 exit143確認後、鏡geometryにMeshBVHを追加し同一形状/条件で再実行。単なる観測timeoutを理由とする再起動ではない。
現行session96324 live。直近出力はthickness0.2/head surfaceDistance1.3244826584mm、witness[40.4332454,0.8138051,252.8988644]。faceおよび1/2/3mmは続行中。次はsame96324をpoll。jsonは各厚のhead+faceが終わってから保存する仕様。
並行してmain-mirror-backing-clearance.mjs/jsonを作成。局所u,v各−40/−20/0/20/40、backdepth5/8/12、75点全てhead最近接面距離>3mmかつ前面から裏点への法線segmentがheadwallを横切らない。R3sphere候補として空間あり。最良u0v0depth5距離44.3666mm。全体のcavity所属/face干渉/保持/剛性は独立未証明で、mirror全厚が通るという意味ではない。公開変更なし。前回verified wait、今回progress。goal active、pending96324のみ。


### 主鏡厚み結果・三角裏枠
前回progress（計算高速化とhead側支持点）。今回same96324をpollして全結果/exit0を確認。main-mirror-thickness.json：厚0.2/1/2/3mmでhead最小表面距離1.324483/0.884861/0.327705/0mm。3mmは接触交差、採用不可。全厚face距離17.694081mm。2mmは名目非接触だが実物公差の余裕が小さい。主鏡前面・輪郭・公開形状は変更なし。
main-mirror-backing-face-clearance.mjs/jsonで75sphere候補がfaceに対しても距離>3、frontからbackへの法線経路非横断。main-mirror-back-frame.mjs/jsonで鏡local[-20,-20,-8],[20,-20,-8],[0,20,-8]の三角R3部材を定義。0.5mm空間サンプル+0.25001guardでhead/face面隙間正。未接続候補で、鏡/頭への取り付け・強度・光束全体・material内包判定は別途。支持枠公開未追加。
次は主鏡実厚と公差余裕：反射前面を保った背面縁形状（例えばbevel、製作可能性未確認）または許可範囲内の内壁余裕を比較。3mmそのまま合格としない。既存の表面距離だけで実機収納完成とはしない。全process終了pendingなし、goal active、progress。


### 2026-09-22 小型男性胸像への外観変更
ユーザーが幅430mmの筋肉質案を大きすぎ・人体に見えないと指摘。頭部は変えず、上腕を省いた短い胸像へ更新。新外形330×211×220mm（幅×奥行×高）、旧430×336.55×278mm比で幅23%、奥行37%、高さ21%減。断面を楕円に変更、胸筋・胸骨溝・鎖骨・上背部の起伏、首へ傾斜する肩を生成。寸法は造形目標で成人の実測値ではない。
work/buildable-bust-v1/generate-wide-backup.mjsに直前ジェネレータ保存。現行generate.mjsから8分割STL生成、全8部品で辺接続数2、仮定220×220×250mm以内。ただし継手・ポート・正常厚み・可動干渉は未検証。buildable-bust-study.htmlを再読込しスクリーンショットで確認。旧モーター寸法枠が底より突出し、旧機構の収納は成立していない。外装縮小だけで達成とは扱わない。プロジェクタや光学系を縮小していない。
暫定モーターXM430-W350-T：28.5×46.5×34mm、82g、12V、停動4.1Nmは連続許容値ではない。公式 https://en.robotis.com/shop_en/item.php?it_id=902-0124-000 。質量・重心・連続トルクが未確定のため購入推奨確定前。TI G2は19V電源案、12Vサーボと分離。HDMI/DC背面パネルは位置表示のみ。
次工程は小型胸像へ実寸プロジェクタ・ケーブル・モーターを再配置し干渉を解消。主鏡厚2mm公差不足、可動首クリアランス、放熱、接合設計を継続。goal active、完成扱い不可。

## 2026-09-22 Budget mechanism and user correction: neck alignment
- User prioritizes overall assembly design, explicitly pauses slicing, caps motors/mirrors/etc at JPY30,000. ST3215 12V x3 and Waveshare Bus Servo Adapter(A) replace costly XM430 candidate. Budget allocations total30k, not confirmed purchase prices; mirror quotes outstanding. No orders sent. 3x2.7A stall exceeds adapter5A; distribution/protection unresolved.
- New public budget-system-study.js and main buildable-bust-study.html created; prior page backed up under work/buildable-bust-v2/pre-budget-view.html. Official manufacturer ST3215 STEP exported to publicSTL; motor-CAD datum established. Existing installed tools only.
- Historic candidate pivot[35,0,-65], torsoShiftX15 passed discrete motor and pulley screens (21 combinations in selected-drive-layout.json). Those results are now INVALID for new neck axes.
- User noticed head/torso neck offset. Extracted original-head outer section at Z42.6055: center[-22.01617,.345745,42.60553], size110.493x119.749, maximum angular interpolation gap1.3654deg. Prior torso neck center30.7629 was52.779mm too far forward.
- Added neck-datum.py, neck-surface.mjs, build-neck.mjs, check-aligned-neck.mjs. Public neck-datum.json, aligned-neck binary geometry, aligned-neck-screen.json, NECK_ALIGNMENT.md.
- Head/face original exterior unchanged. Rigid chest belowZ-20 unchanged. Upper flexible cover transitions to actual head section (0.5mm radial inset overlap). UI rotates about neck section center, dynamically interpolates flexible-cover endpoints; not material simulation.
- New anatomical-datum pivot makes old45/20/15 range FAIL projector/chest containment screen (negative58.12mm planar section witness).15/5/5 sample27poses positive4.37mm; neutral16.86. Positive vertex screen NOT full collisionproof; deformingcover/motors/cables/optics excluded. UI small range explicitly provisional, not reduced goal acceptance. Oldmotors hiddenbydefault and static retiredreference only. Old8.43mm headgap claim withdrawn.
- Main UI now opaque by default so neck alignment is visible. Must not claim printable assembly/whole-face optical success. Budget and neck mechanism remain engineering work in progress.

## 2026-09-22 User rejects stretched neck; clean Furhat-like exterior rebuild
- User explicitly rejected the long curved neck as unnatural. Rebuilt whole bust exterior, not just the neck transition. Main page updated and visually inspected.
- New generator work/buildable-bust-v3/generate.mjs; public assets/buildable-bust-v3/{body,collar,panel} binary meshes, design.json, DESIGN.md, neutral-screen.json. Previous main JS+HTML backed up in work/buildable-bust-v3.
- Short collar18.606mm (Z24..42.606), raised shoulder surface, width300mm, bodyheight184mm. Smooth white exterior, muted dark front servicepanel concept. Removed muscular lumps and visual shellsplit lines. Head/face unchanged. Added front/side/oblique camera buttons.
- Initial206mm-depth variant FAILED neutral projector fit (19.68mm beyond ellipse screen); corrected lowerchest depth/radius to max232mm and shifted lowersectioncenters slightlyforward, keeping neckdatumfixed. Neutral309hullvertexscreen with3mm radialwallallowance nowminimum5.643mm conservativeellipseindicator. Not Euclidean clearance/fullcontainment proof. No new motion/motor/optical validation.
- Old mechanism remains retired staticreference, hiddenbydefault. Budget30k unchanged, no orders/dependency installs/slicing. Current collar surface is open concept, not printablefinishedpart.
- Official Furhat product page and official technicaloverview indexeddescription researched. Source linked in publicdesignnote. Exterior direction is smooth understated bust emphasizing face.

## 2026-09-22 Neck/internal redesign v4 requested by user
- Preserved approvedv3 exterior/head/face. Implemented new moving mechanism in current budget-system-study.js and mainHTML. New files under work/buildable-bust-v4 and publicassets/buildable-bust-v4; prior JS/HTML backed up there.
- New common pivot[-22.0161686,.3457451,20],22.606mm below neckseam. Original seam-level pivot failed ring/head fit. Moving lower solved sampled ring/head/body surface intersections without elongating exterior.
- Candidate operating screen yaw±30,pitch0..12 (forwardonly),roll±6. Backwardtilt unresolved. motion-audit.json:10309poses1deglattice,309projectorhullvertices,3mmradialwallallowance,minellipseindicator4.7684mm. This is not continuous/volume collisionproof.
- Head/face versus actualv3body27poses hits0. Three hollowring STL conceptpartsyaw OD100ID82th6z-20,pitchOD100ID92th6z20,rollOD90ID82th6z20. Rings27poses surfacehits0 head/face/body; sampledchiefrays0blocks. Finitepupilnotproven.
- Motorsearch iterated center48.3/80/100andshaftdirections. SelectedST3215 officialCAD OBBmotors afterbody/projector/mutual andoptical+4mmenvelopechecks.336passingcombinations, selected minmotorskinscreen10.816mm. Pitchaxispoints-Y,roll+X,yaw+Z. New publicselected-drive-layout.json onlycurrentcandidate. Oldv2 retired.
- Pitch/roll3:1 belt centers100mm, tangentsphysicallycomputed (box2x6mm).4spans27posesprojectorintersection0,chief-rayinflated4mm0. Beltarcs/otherhardware/continuousnotvalidated. Possible280mmbeltrequires99.2mmcenteradjustmentandrerun.
- Yawconcept150T/20T7.5ratioannularplainbearing; gearedteeth/bearingseatsanti-liftnotdesigned.2pitch+2roll608bearingsandtwoyawpostsrenderedplacementonly,notcollisionverified. RawringSTLsNOTload-ready; ring6mmthickrequiresbearingbossesfor8mmshafts. Fullhead/projectorloadframeunresolved.
- Public MECHANISM.md recordsarchitecture,assemblysequence,torqueexamplewithunknownmass,HDMI/DCstrainrelieffuturework,adapter5A vs3stall8.1A issue (separateservopowerproposed,notfinalwiring). Budget30kstillallocationonly;noorders,noinstalls,noslicing.
- Currentwebmanufacturer sourcesverified ST3215wiki30kgcm12V/2.7A/45rpm; adapterFAQ5A. No claimcontinuousratedtorque or manufacturedassemblysuccess.

### v4 FINAL REVISION (supersedes initial v4 geometry details above)
- Extra surface tests revealed original Ø44x10 pulleys collide with head/body; rejected. Reduced pitch/roll to40T:20T2:1, pulley envelopeØ28x6. Output centers pitch[-22.01617,51,20],roll[12,.345745,20]. Same100mm nominalmotorcenter distance,260mm belt~99.8mm adjustedcenter (notyetvalidated).
- Changed4trunnionbearings to688ZZ8x16x5; bearingcenterspitchY±45androllX±43relativepivot; yawpostsY±45. InnerrollcarrierOD82ID74th6. Rings,headbodychecksre-run:0sampledsurfacehits,chief-ray0blocks. Extras-surface-screen.json all8items (2pulleys,4bearingcylinders,2posts)0head/face/bodyintersectionat27poses. Othermechanismpairs/finiteopticsremaining.
- Refinedmotoranglegridphi5degpsi15deg, opticalenvelopefilterthenpaircompatibilitycache;141? CORRECTION exact passingCombinations=14004. Selectedbodyminscreen9.953624mm. Allmotorcasesrequire>=3mmnominalvertexmargin. AddeddrivenpulleyOBBpairassert;passes. Finalselectedposes yawphi210psi75,pitchphi260psi345,rollphi220psi0. DetailedofficialCADtip/quaternioninpublicJSON.
- Rechecked4beltstraightspanswithnewmotorpositions:27posesprojector0,chief-ray+4mmmargin0. Spansrenderedrectangular2x6insteadoldsketchlines. Mainpagecurrentdataallv4,interiorcheckedbydefault,bodyunchanged.
- PublicBOMv4updatesbearingtypeandpoweruncertainty; allocationsstill30000. MECHANISM.mdrewritten2:1/688/74mmaperture/actualtestlimits. LatestNSK688ZZdimensionsfrommanufacturerreferenceverified. Noorder/install/slicing.

## 2026-09-22 User requests gears for rotation: yaw gear v5
- Interpreted rotation as left/right yaw and explicitly told user. Pitch/roll belts unchanged.
- Built actual involute-profile96T/20T spur gears:module1,20degpressure,width6,center58,ratio4.8,ringOD98ID82,pinionOD22pilotbore5 (NOTservocompatibleadapter). Tooththicknessreduction.1each nominaltotalbacklash.2. Simplifiedroot(no trochoidfillet), prototypeonly. RawlocalSTLs andgear-layout.json inpublicassets/buildable-bust-v5.
- 181phasesoveronepiniontoothperiod usingbothBVHs:0trianglesurfaceintersections. Notcontinuous/contact/strengthproof. Gearmathandmeshingdisplayimplemented;pinionrotationphase-4.8*yaw.
- Yawmotortip moved~3.63mm radiallyoutward to[-72.2456434,29.345745,-20]. Parentorientation unchanged. publicdrive-layout.json supersedesv4motor0tip; otheraxesunchanged. body/headexternalgeometryunchanged.
- 27poses packaging-check:gearvsprojector,yawmotorvsprojectorandothermotors0hits. Yawmotorvertexellipseindicator11.089mm;gears14.421mm with3mmradialwallallowance. Updatedmotor+4mmenvelopevs sampledchief rays passes27poses. Otherhardwareinterfacesnotverified.
- MainHTML/JS nowfetchv5drive andactualgearSTLs;oldyawblankringremoved;pinionplaceholdercylinderremoved. MainHTMLaddsgearspec+YAW_GEARS.md. Pre-v5HTML/JS backupsunderwork/buildable-bust-v5.
- Supersedespreviousconcept150T/20T7.5ratio (whichwouldneed225degmotorfor30degyaw). New4.8requires144deg eachdirection. Servohornadapter/retainers/framebearingmounts/strengthstillunresolved. Noorders/install/slice.


## 2026-09-22 v6 yaw coupling / retention detail
- Official horn STEP: OD19.2, four Ø2.5 holes PCD14; thread not proven by STEP. M3×0.5 incoming inspection gate.
- 20 valid single-solid STEP/STL parts, assembly STEP. 10mm gears, 9.5mm overlap, big bore76 / sleeve72. Horn flange+smooth journal, independent captive M3×30 bolt, calibrated0.08±0.01Nm clutch.
- Lowered yaw servo tipZ−30.4. Upper/lower3mm aluminium capture plates, sixM3×25 countersunk screws inserted from below, spacers11.3. Base reshaped and thinned after projector collisions.
- v6 geometric checks: 27pose part/projector and motor hull checks,181gear phases, selected selfpairs,hardware envelopes, sampledchief rays. See assets/buildable-bust-v6. Not physical proof or all-parts continuous containment.
- Strength calculation is conditional, PETG8MPa and aluminium80MPa are assumed allowables; gear nominal margin1.14 only. Mass2kg/CG50mm/3x load envelope assumed, unmeasured.
- NOT finished assembly: old yawposts conflict with retainingplate; removed from renderer, uppercarrier transition is not designed. Servo mountingframe, exactspring/friction SKUs, actual hornthread and proof test still unresolved. No orders or installs.
