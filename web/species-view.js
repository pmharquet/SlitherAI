(function(root){
  'use strict';
  const color=id=>`hsl(${(Number(id)*137.508+25)%360} 68% 65%)`;
  const number=(n,d=1)=>Number.isFinite(n)?n.toLocaleString('fr-FR',{maximumFractionDigits:d}):'—';
  class SpeciesView{
    constructor(panel){
      this.panel=panel;this.canvas=panel.querySelector('canvas');this.history=[];
      new ResizeObserver(()=>this.draw()).observe(this.canvas);
      this.canvas.onmousemove=event=>{
        if(!this.history.length)return;
        const r=this.canvas.getBoundingClientRect(),i=Math.max(0,Math.min(this.history.length-1,Math.floor((event.clientX-r.left-42)/(r.width-60)*this.history.length)));
        const row=this.history[i];
        this.canvas.title=`Génération ${row.generation+1} · ${row.count} espèces · ${row.population} réseaux\n`+row.rows.map(s=>`E${s.id} : ${s.size}`).join(' · ')+(row.reconstructed?'\nEffectifs récupérés depuis la sauvegarde':'');
      };
    }
    update(state){
      const data=state.species||{},current=data.current,last=data.latest,params=data.parameters||{};
      const protocol=state.status?.protocol,selfplay=protocol?.opponent_mode==='selfplay';
      this.history=state.species_history||[];
      const text=(id,value)=>this.panel.querySelector('#'+id).textContent=value;
      if(!current){text('species-stamp','Les détails apparaîtront à la prochaine génération.');return;}
      text('species-stamp',`Population G${current.generation+1} · ${current.population} réseaux distincts`);
      text('species-active',current.count);
      text('species-effective',number(current.effective));
      text('species-dominant',number(current.largest_share*100)+' %');
      text('species-turnover',last?`+${last.new_ids.length} / −${last.removed_ids.length}`:'—');
      text('species-turnover-note',last?`Transition G${last.generation+1} → G${last.generation+2}`:'Naissances / disparitions');
      const champion=last?.rows.reduce((a,b)=>!a||b.best>a.best?b:a,null);
      text('species-champion',champion?`E${champion.id} · ${number(champion.best)}`:'—');
      text('species-champion-note',last?`Meilleure fitness · G${last.generation+1}`:'Dernière évaluation terminée');
      text('species-rules',`${params.target_min?`Objectif ${params.target_min}–${params.target_max} espèces · `:''}Seuil génétique ${number(params.compatibility_threshold,3)} · ${params.progress_window?`${selfplay?'score de sélection coévolutif':'moyenne fixe'}, fenêtres de ${params.progress_window} générations · `:''}stagnation ${params.max_stagnation} générations · ${params.species_elitism} espèces protégées · ${params.elitism} élites par espèce · parents : meilleurs ${number(params.survival_threshold*100)} % (au moins 2)`);
      const bar=this.panel.querySelector('#species-composition');bar.replaceChildren();
      const legend=this.panel.querySelector('#species-legend');legend.replaceChildren();
      for(const row of current.rows){
        const slice=document.createElement('span');slice.style.width=(row.size/current.population*100)+'%';slice.style.background=color(row.id);slice.title=`Espèce ${row.id} : ${row.size} réseaux`;bar.append(slice);
        const tag=document.createElement('span');tag.style.setProperty('--species-color',color(row.id));tag.textContent=`E${row.id} · ${row.size}`;legend.append(tag);
      }
      const config=state.status?.config,replicates=selfplay?protocol.games_per_genome:protocol?protocol.anchor_games+protocol.rotating_games:config?config.maps*config.worms/state.status.population:null;
      text('species-table-stamp',last?`Évaluation G${last.generation+1}${replicates?` · ${replicates} parties par réseau${selfplay?' · adversaires coévolutifs':protocol?' · 80 % référence fixe + 20 % nouvelle carte':''}`:''}`:'En attente d’une évaluation complète');
      const anchorHeader=this.panel.querySelector('thead th:nth-child(11)');
      anchorHeader.textContent=selfplay?'Score sélection':'Score fixe';
      anchorHeader.title=selfplay?'Moyenne des scores de sélection coévolutifs, utilisée pour la stagnation':'Moyenne des scores agrégés sur les quatre parties fixes, utilisée pour la stagnation';
      this.panel.querySelector('thead th:nth-child(9)').title=selfplay?'Générations sans progrès du score de sélection coévolutif':'Générations sans progrès entre fenêtres de cinq générations du score moyen sur les parties fixes';
      // Rebuild only after a generation ends; keep text selectable between updates.
      const signature=JSON.stringify(last);
      if(this.signature!==signature){
        this.signature=signature;const body=this.panel.querySelector('tbody');body.replaceChildren();
        for(const row of [...(last?.rows||[])].sort((a,b)=>b.best-a.best)){
          const tr=document.createElement('tr');tr.dataset.species=String(row.id);
          const values=[`E${row.id}`,`${row.size} → ${row.next_size}`,number(row.best),number(row.mean),number(row.adjusted_fitness,3),row.age,number(row.hidden),number(row.connections),`${row.stagnant_for} / ${params.max_stagnation}`,row.removed?'Éliminée':row.protected?'Protégée':'Active',number(row.anchor_mean),number(row.food_gain),Number.isFinite(row.survival)?number(row.survival*100)+' %':'—'];
          values.forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i===0){td.style.color=color(row.id);td.className='species-id';}if(i===8){td.title=`Dernière progression : génération ${row.last_improved+1}${params.progress_window?` · médiane récente ${number(row.recent_score)}`:''}`;const meter=document.createElement('div');meter.className='stagnation-meter';meter.style.setProperty('--progress',Math.min(100,row.stagnant_for/params.max_stagnation*100)+'%');td.append(meter);}if(i===9)td.className=row.removed?'species-removed':row.protected?'species-protected':'';tr.append(td);});
          body.append(tr);
        }
      }
      text('species-history-stamp',this.history.some(r=>r.reconstructed)?'Effectifs anciens récupérés des sauvegardes ; fitness détaillées enregistrées depuis cette mise à jour.':'Effectifs par espèce et par génération.');
      this.draw();
    }
    draw(){
      const canvas=this.canvas,r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;
      canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);
      const ctx=canvas.getContext('2d');ctx.setTransform(d,0,0,d,0,0);const w=r.width,h=r.height;
      ctx.clearRect(0,0,w,h);if(!this.history.length)return;
      const max=Math.max(...this.history.map(r=>r.population)),plotH=h-42,barW=(w-60)/this.history.length;
      ctx.font='11px system-ui';ctx.fillStyle='#91a6b5';ctx.fillText(String(max),8,18);ctx.fillText('0',20,h-26);
      this.history.forEach((row,i)=>{let y=h-30;const x=42+i*barW;for(const s of row.rows){const sh=s.size/max*plotH;ctx.fillStyle=color(s.id);ctx.fillRect(x,y-sh,Math.max(.5,barW-1),sh);y-=sh;}if(i===0||i===this.history.length-1||i%Math.max(1,Math.ceil(this.history.length/8))===0){ctx.fillStyle='#91a6b5';ctx.fillText(String(row.generation+1),x,h-9);}});
    }
  }
  root.SpeciesView=SpeciesView;root.speciesColor=color;
})(window);
