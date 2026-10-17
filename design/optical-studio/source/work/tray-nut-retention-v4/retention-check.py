exec(open('work/tray-nut-retention-v4/cad-mates.py').read().split('tray=read')[0])
cap=read(out/'tray-nut-cover-local.step');seat=read(out/'tray-nut-fit-coupon.step');nut=cut(hex_y(0,-6.95,0,2.4,5.5),cy(1.3,[0,-7.05,0],[0,1,0],2.6))
def translate(s,x,y,z):return move(s,[1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1])
checks={'nutDownward0_7mmBlockedVolumeMM3':volume(BRepAlgoAPI_Common(translate(nut,0,-.7,0),cap).Shape()),'coverDownward0_4mmBlockedVolumeMM3':volume(BRepAlgoAPI_Common(translate(cap,0,-.4,0),seat).Shape()),'coverWithdrawal1mmBlockedVolumeMM3':volume(BRepAlgoAPI_Common(translate(cap,0,0,1),seat).Shape()),'scope':'Rigid geometric blocking only; elastic snap insertion and retention forces require actual printed test.'}
print(checks);assert all(checks[k]>.01 for k in list(checks)[:3]);(out/'retention-check.json').write_text(json.dumps(checks,indent=2))
