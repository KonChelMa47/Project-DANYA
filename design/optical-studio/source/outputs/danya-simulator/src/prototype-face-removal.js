// Fixed-prototype face unit: a straight forward service path, not a moving optical mode.
export function installFaceRemoval({meshes, camera, controls, rays}) {
 const panel=document.createElement('details');panel.open=false;
 panel.innerHTML=`<summary>v12：顔ユニットの着脱</summary><p>顔を支えながら首開口から支持枠の裏を押し、顔＋支持枠を前へ外す設計です。外側に取っ手はありません。実物での指の届き方は要確認です。</p><label>引き出し量 <output id="face-removal-distance">0</output> mm <input id="face-removal" type="range" min="0" max="100" step="1" value="0"></label><br><button id="face-removal-close">組立位置に戻す</button> <button id="face-removal-open">取り外した状態</button><p>着脱途中の干渉は離散位置で確認。実物の着脱力・接着強度は未検証です。</p><a href="assets/projection-prototype/face-v12/ASSEMBLY.md">着脱・交換手順</a> · <a href="assets/projection-prototype/face-v12/face-v12-update-print.zip">交換する3部品のSTL</a>`;
 document.querySelector('aside').prepend(panel);
 const slider=document.getElementById('face-removal');
 function move(x){const projection=document.getElementById('face-projection');if(projection?.checked){projection.checked=false;projection.dispatchEvent(new Event('change'));}const ceiling=document.getElementById('ceiling-on');if(ceiling?.checked){ceiling.checked=false;ceiling.dispatchEvent(new Event('change'));}
  for(const m of Object.values(meshes))m.position.set(0,0,0);
  for(const[name,m]of Object.entries(meshes)){if(name==='face'||name==='magnet-carrier'||/^magnet-clear-\d+-face-magnet$/.test(name)){m.visible=true;m.position.set(x,0,0);}}
  meshes.head.visible=true;meshes['head-upper'].position.set(0,0,0);meshes['head-lower'].position.set(0,0,0);
  document.getElementById('face').checked=true;document.getElementById('head').checked=true;document.getElementById('explode-face').checked=false;
  const rayToggle=document.getElementById('rays');rayToggle.disabled=x!==0;rays.visible=x===0&&rayToggle.checked;
  slider.value=x;document.getElementById('face-removal-distance').value=x;
 }
 slider.oninput=()=>move(Number(slider.value));document.getElementById('face-removal-close').onclick=()=>move(0);
 document.getElementById('face-removal-open').onclick=()=>{move(80);camera.fov=38;camera.updateProjectionMatrix();camera.position.set(410,-540,260);controls.target.set(20,0,157);controls.update();};
 document.getElementById('face-projection')?.addEventListener('change',()=>{if(document.getElementById('face-projection').checked){slider.value=0;document.getElementById('face-removal-distance').value=0;document.getElementById('rays').disabled=false;}});
}
