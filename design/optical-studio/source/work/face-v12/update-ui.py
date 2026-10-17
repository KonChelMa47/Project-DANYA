from pathlib import Path
import json
r=Path('outputs/danya-simulator');paths=['projection-prototype.html','src/projection-prototype.js','src/prototype-face-removal.js']
backup=Path('work/face-v12/ui-before');backup.mkdir(exist_ok=True)
for name in paths:
 p=r/name;(backup/p.name).write_bytes(p.read_bytes());s=p.read_text().replace('face-v11','face-v12').replace('v11','v12').replace('STATIC PROTOTYPE 11','STATIC PROTOTYPE 12')
 if name.endswith('.html'):
  s=s.replace('v12：あご下のつまみで顔＋支持枠を前へ外せます。上側磁石を鏡の前方へ移設。','v12：露出したつまみをなくし、磁石を滑らかな周縁の内側へ収めました。上側はØ3×1 mm、下側は従来のØ5.9×2.9 mmです。')
  s=s.replace('<details open>','<details>')
  s=s.replace('顔側4か所・後頭部側4か所をそれぞれ本体と一体化。磁石Ø5.9×2.9 mmは、着座ストッパー＋接着剤用の溝を備えたポケットへ二液エポキシで固定します。','上側は顔側2個・後頭部側2個のØ3×1 mm磁石、下側は顔側2個・後頭部側2個のØ5.9×2.9 mm磁石です。ポケットに二液エポキシで固定します。')
 elif name.endswith('prototype-face-removal.js'):
  s=s.replace('panel.open=true','panel.open=false').replace('あご下の2つのつまみを持ち、顔＋支持枠をまっすぐ前へ引き出します。上側の磁石は鏡より前に配置しました。','顔を支えながら首開口から支持枠の裏を押し、顔＋支持枠を前へ外す設計です。外側に取っ手はありません。実物での指の届き方は要確認です。')
 else:
  s=s.replace("['face','magnet-carrier'].includes(p.name)?0xe5d8c1:['head-upper','head-lower'].includes(p.name)?0xdce3dd:","['face','magnet-carrier','head-upper','head-lower'].includes(p.name)?0xe3e5df:")
 p.write_text(s)
p=r/'assets/projection-prototype/face-preview/metadata.json'
if p.exists():
 d=json.loads(p.read_text());d['verifiedRevision']='face-v12';p.write_text(json.dumps(d,indent=2))
