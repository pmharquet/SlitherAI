function updateLearning(state){
  const status=state.status||{},progress=status.progress||{},last=status.last_metrics||state.history?.at(-1)||{};
  const metrics=last.evaluation||{},format=(x,d=1)=>Number.isFinite(x)?x.toLocaleString('fr-FR',{maximumFractionDigits:d}):'—';
  const put=(id,text)=>document.getElementById(id).textContent=text;
  const percent=x=>Number.isFinite(x)?format(x*100)+' %':'—';
  put('training-progress',progress.games?`Partie ${progress.game}/${progress.games} (${progress.anchor?'référence fixe':'nouvelle carte'}) · lot ${progress.batch}/${progress.batches} · ${progress.completed_episodes}/${progress.total_episodes} parties terminées`:'Préparation des parties comparables');
  const bar=document.getElementById('training-progress-bar');bar.max=progress.total_episodes||1;bar.value=progress.completed_episodes||0;
  put('behavior-stamp',last.evaluation?`Moyennes de la population · G${last.generation+1}`:'Détails à la fin de la première génération');
  put('food-kpi',format(metrics.food_gain));put('survival-kpi',percent(metrics.alive));
  put('border-kpi',percent(metrics.border_death));put('collision-kpi',percent(metrics.collision_death));
  put('boost-kpi',percent(metrics.boost_fraction));put('turn-kpi',format(metrics.turn_degrees)+'° / décision');
  put('spread-kpi',format(metrics.episode_score_sd));
  put('reward-kpi',`Croissance ${format(metrics.reward_growth)} · survie ${format(metrics.reward_survival)} · kills ${format(metrics.reward_kills)} · mort ${format(metrics.reward_death)}`);
  const validation=[...(state.history||[])].reverse().find(row=>row.validation)?.validation;
  put('validation-details',validation?`${validation.maps||4} arènes · ${validation.seconds} s · survie ${percent(validation.alive)} · nourriture ${format(validation.food_gain)}`:'Validation : 32 arènes indépendantes de 90 s');
  put('baseline-kpi',`Heuristique ${format(state.baselines?.heuristic?.fitness)} · cercle ${format(state.baselines?.circle?.fitness)}`);
}
