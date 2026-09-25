const $=id=>document.getElementById(id);let state={},snapshot=null,selected=0,history=[];
let selectedRun=new URLSearchParams(window.location.search).get('run_id')||'';
let lastRunListRefresh=0;
const labels={starting:'Initialisation',training:'Entraînement',validating:'Validation sur des arènes fixes',paused:'En pause',stopped:'Arrêté · sauvegarde disponible',completed:'Session terminée',interrupted:'Interrompu · reprise disponible',error:'Erreur'};
const speciesView = new SpeciesView($('species-panel'));
const networkView = new NetworkView($('network-panel'), worm => { if(!state.read_only) command('/api/control', {network_worm: worm}); });
async function api(path,data){const r=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const j=await r.json();if(!r.ok)throw Error(typeof j.detail==='string'?j.detail:JSON.stringify(j.detail));return j;}
async function command(path,data){$('error').textContent='';try{await api(path,data);await refresh();}catch(e){$('error').textContent=e.message;}}
function options(resume=false){return{maps:+$('maps-count').value,worms:+$('worms').value,population:+$('population').value,generations:+$('generations').value,seconds:+$('seconds').value,device:$('device').value,resume};}
function runUrl(path){return selectedRun?`${path}?run_id=${encodeURIComponent(selectedRun)}`:path;}
function chooseRun(value){selectedRun=value;const url=new URL(window.location.href);if(selectedRun)url.searchParams.set('run_id',selectedRun);else url.searchParams.delete('run_id');window.history.replaceState({},'',url);refresh();}
async function refreshRunList(){
  const entries=await api('/api/runs');
  const picker=$('run-picker');
  const optionsList=[new Option('Run du tableau (défaut)','')];
  for(const entry of entries){
    const generation=entry.generation==null?'':` · G${entry.generation+1}${entry.target_generation!=null?`/${entry.target_generation}`:''}`;
    const phase=labels[entry.phase]||entry.phase||'état inconnu';
    const mode=entry.opponent_mode?` · ${entry.opponent_mode}`:'';
    optionsList.push(new Option(`${entry.name} · ${phase}${generation}${mode}`,entry.name));
  }
  if(selectedRun&&!entries.some(entry=>entry.name===selectedRun)){
    selectedRun='';const url=new URL(window.location.href);url.searchParams.delete('run_id');window.history.replaceState({},'',url);
  }
  picker.replaceChildren(...optionsList);picker.value=selectedRun;lastRunListRefresh=Date.now();
}
$('run-picker').onchange=()=>chooseRun($('run-picker').value);
$('start').onclick=()=>command('/api/start',options());$('resume').onclick=()=>command('/api/start',options(true));$('stop').onclick=()=>command('/api/control',{stop:true,pause:false});$('pause').onclick=()=>command('/api/control',{pause:state.status?.phase!=='paused'});
async function refresh(){
  try{
    if(Date.now()-lastRunListRefresh>10000)await refreshRunList();
    state=await api(runUrl('/api/state'));
    const s=state.status||{},m=s.last_metrics||state.history?.at(-1)||{};history=state.history||[];
    const phaseLabel=labels[s.phase]||'Prêt · entraînement arrêté';
    if(state.read_only){const age=state.status_age_seconds==null?'âge inconnu':`statut il y a ${Math.round(state.status_age_seconds)} s`;$('phase').textContent=`${phaseLabel} · lecture seule · processus non suivi (${age})`;$('phase').classList.add('readonly-note');}
    else{$('phase').textContent=phaseLabel;$('phase').classList.remove('readonly-note');}
    $('generation').textContent=s.generation!=null?`${s.generation+1} / ${s.target_generation}`:'—';$('alive').textContent=s.alive??'—';$('fitness').textContent=m.best?.toFixed(1)??'—';$('throughput').textContent=s.agent_steps_per_second?.toLocaleString('fr-FR')??'—';
    for(const key of ['species','nodes','connections'])$(key).textContent=typeof m[key]==='number'?m[key].toFixed(key==='species'?0:1):'—';
    const v=[...history].reverse().find(x=>x.validation);$('validation').textContent=v?v.validation.fitness.toFixed(1):'—';
    const readOnly=!!state.read_only;$('start').disabled=readOnly||state.active;$('resume').disabled=readOnly||state.active||!state.can_resume;$('pause').disabled=readOnly||!state.active;$('stop').disabled=readOnly||!state.active;$('network-worm').disabled=readOnly;
    $('pause').textContent=s.phase==='paused'?'Continuer':'Pause';if(s.gpu)$('gpu').textContent=`${s.gpu} · ${s.gpu_memory_mb??'—'} Mo utilisés`;if(state.error)$('error').textContent=state.error;
    snapshot=await api(runUrl('/api/preview'));
    $('network-worm').disabled=readOnly||!snapshot;
    if(snapshot){selected=snapshot.arena;$('empty').style.display='none';const previewAge=snapshot.monitoring?.preview_age_seconds;const captureAge=previewAge==null?'':` · capture il y a ${Math.round(previewAge)} s`;$('time').textContent=`Temps simulé ${snapshot.elapsed.toFixed(1)} s${readOnly?' · aperçu lecture seule'+captureAge:''}`;$('arena-title').textContent=`Arène ${snapshot.arena+1}${readOnly?' · capture du run sélectionné':''}`;
      if($('maps').children.length!==snapshot.maps.length){$('maps').replaceChildren(...snapshot.maps.map(m=>{const b=document.createElement('button');b.onclick=()=>{if(!state.read_only){selected=m.id;command('/api/control',{arena:m.id});}};return b;}));}
      snapshot.maps.forEach((m,i)=>{const b=$('maps').children[i];b.textContent=`${i+1} · ${m.alive}`;b.className=m.id===selected?'selected':'';b.disabled=readOnly;});
    }else{selected=0;$('empty').style.display='grid';$('empty').querySelector('strong').textContent='Aucun aperçu disponible';$('empty').querySelector('p').textContent=state.read_only?'Ce run n’a pas encore enregistré de capture d’arène.':'La session n’a pas encore publié d’aperçu.';$('empty').querySelector('small').textContent=state.read_only?'Le run sélectionné reste consultable en lecture seule.':'L’arène apparaîtra au prochain aperçu.';$('time').textContent='Temps simulé —';$('arena-title').textContent='Aucune arène capturée';$('maps').replaceChildren();}
    networkView.update(snapshot,state);speciesView.update(state);updateLearning(state);if(state.species?.current)$('species').textContent=state.species.current.count;draw();drawChart();
  }catch(e){$('error').textContent='Serveur local indisponible : '+e.message;}
}
function size(canvas){const d=window.devicePixelRatio||1;const r=canvas.getBoundingClientRect();if(canvas.width!==Math.round(r.width*d)||canvas.height!==Math.round(r.height*d)){canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);}const ctx=canvas.getContext('2d');ctx.setTransform(d,0,0,d,0,0);return[ctx,r.width,r.height];}
function draw(){const[ctx,w,h]=size($('world'));ctx.clearRect(0,0,w,h);if(!snapshot)return;let cx=0,cy=0,scale=Math.min(w,h)/(snapshot.radius*2.12);const live=snapshot.worms.filter(x=>x.alive&&x.body.length);if($('view').value==='follow'&&live.length){const leader=snapshot.worms.find(w=>w.controller==='neat'&&w.body.length)||live.reduce((a,b)=>a.mass>b.mass?a:b);[cx,cy]=leader.body[0];scale=Math.min(w/1400,h/1000);}ctx.save();ctx.translate(w/2,h/2);ctx.scale(scale,scale);ctx.translate(-cx,-cy);ctx.strokeStyle='#1c2b36';ctx.lineWidth=1/scale;const grid=150;for(let x=Math.floor((cx-w/scale/2)/grid)*grid;x<cx+w/scale/2;x+=grid){ctx.beginPath();ctx.moveTo(x,cy-h/scale/2);ctx.lineTo(x,cy+h/scale/2);ctx.stroke();}for(let y=Math.floor((cy-h/scale/2)/grid)*grid;y<cy+h/scale/2;y+=grid){ctx.beginPath();ctx.moveTo(cx-w/scale/2,y);ctx.lineTo(cx+w/scale/2,y);ctx.stroke();}ctx.beginPath();ctx.arc(0,0,snapshot.radius,0,Math.PI*2);ctx.strokeStyle='#8155aa';ctx.lineWidth=4/scale;ctx.stroke();for(let i=0;i<snapshot.food.length;i++){const[x,y,s]=snapshot.food[i];if(Math.abs(x-cx)>w/scale/2+20||Math.abs(y-cy)>h/scale/2+20)continue;ctx.fillStyle=`hsl(${(i*137.5)%360} 80% 67%)`;ctx.beginPath();ctx.arc(x,y,Math.max(s*.75,1.1/scale),0,Math.PI*2);ctx.fill();}for(const worm of live){const reference=worm.controller==='reference',sid=snapshot.species_ids?.[worm.id],hue=reference?205:$('color-mode').value==='species'&&sid!=null?(sid*137.508+25)%360:(worm.id*137.5)%360;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();worm.body.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.strokeStyle=`hsl(${hue} ${reference?20:76}% ${reference?35:51}%)`;ctx.lineWidth=worm.radius*2;if(worm.boost){ctx.shadowColor=`hsl(${hue} 100% 72%)`;ctx.shadowBlur=14;}ctx.stroke();ctx.shadowBlur=0;ctx.lineWidth=worm.radius*.8;ctx.strokeStyle=`hsl(${hue} ${reference?20:88}% ${reference?47:70}%)`;ctx.stroke();const[head,tail]=worm.body;if(tail){const angle=Math.atan2(head[1]-tail[1],head[0]-tail[0]);for(const side of [-1,1]){const ex=head[0]+Math.cos(angle)*worm.radius*.4-Math.sin(angle)*worm.radius*.5*side,ey=head[1]+Math.sin(angle)*worm.radius*.4+Math.cos(angle)*worm.radius*.5*side;ctx.fillStyle='white';ctx.beginPath();ctx.arc(ex,ey,worm.radius*.32,0,Math.PI*2);ctx.fill();ctx.fillStyle='#111922';ctx.beginPath();ctx.arc(ex+Math.cos(angle)*worm.radius*.12,ey+Math.sin(angle)*worm.radius*.12,worm.radius*.16,0,Math.PI*2);ctx.fill();}}}ctx.restore();}
function drawChart(){const[ctx,w,h]=size($('chart'));ctx.clearRect(0,0,w,h);if(!history.length)return;const vals=history.flatMap(x=>[x.best,x.mean]),lo=Math.min(0,...vals),hi=Math.max(lo+1,...vals);ctx.font='11px system-ui';ctx.fillStyle='#8194a3';ctx.fillText(hi.toFixed(0),10,20);ctx.fillText(lo.toFixed(0),10,h-12);for(const[key,color]of[['best','#67e7b9'],['mean','#69aef4']]){ctx.beginPath();history.forEach((v,i)=>{const x=42+i/Math.max(1,history.length-1)*(w-60),y=h-20-(v[key]-lo)/(hi-lo)*(h-40);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}}
$('view').onchange=draw;$('color-mode').onchange=draw;window.addEventListener('resize',()=>{draw();drawChart();});
async function initializeDashboard(){try{await refreshRunList();}catch(e){$('error').textContent=e.message;}await refresh();}
initializeDashboard();setInterval(refresh,1200);
