(function(root) {
  'use strict';
  const channelNames = ['Portée visible', 'Têtes ennemies', 'Corps ennemis', 'Mon corps', 'Bordure', 'Boules / proies'];
  const globalNames = ['Vitesse', 'Rayon du corps', 'Longueur', 'Cap : sinus', 'Cap : cosinus', 'Virage : sinus', 'Virage : cosinus', 'Boost précédent'];

  function inputNode(data, index) {
    const channels = data.schema.ray_channels.length;
    const sensoryCount = data.schema.angles_degrees.length * channels;
    const sensory = index < sensoryCount;
    const group = sensory ? index % channels : channels + index - sensoryCount;
    const angle = sensory ? data.schema.angles_degrees[Math.floor(index / channels)] : null;
    const name = sensory ? `${channelNames[group]} · ${angle > 0 ? '+' : ''}${angle}°` : globalNames[index - sensoryCount];
    return {id: data.input_keys[index], kind: 'input', group, index, angle, label: name, value: data.inputs[index], expressed: true};
  }

  function buildGraph(data, mode='grouped', showDisabled=false) {
    const links = data.connections.filter(e => showDisabled || (e.enabled && e.expressed));
    const visibleUsed = new Set(links.filter(e => e.source < 0).map(e => e.source));
    const used = new Set(data.connections.filter(e => e.enabled && e.expressed && e.source < 0).map(e => e.source));
    const inputs = data.inputs.map((_, index) => inputNode(data, index));
    const nodes = data.nodes.map(n => ({...n}));
    const sourceMap = new Map();
    if (mode === 'grouped') {
      for (let group = 0; group < 6; group++) {
        const members = inputs.filter(n => n.group === group);
        const id = `group-${group}`;
        nodes.push({id, kind: 'group', group, label: channelNames[group], count: members.length,
          connected: members.filter(n => used.has(n.id)).length,
          value: members.reduce((s, n) => s + Math.abs(n.value), 0) / members.length,
          peak: Math.max(...members.map(n => Math.abs(n.value))), expressed: members.some(n => used.has(n.id))});
        members.forEach(n => sourceMap.set(n.id, id));
      }
      inputs.filter(n => n.group >= 6).forEach(n => {nodes.push({...n, expressed: used.has(n.id)}); sourceMap.set(n.id, n.id);});
    } else {
      inputs.filter(n => mode === 'all' || visibleUsed.has(n.id)).forEach(n => {nodes.push({...n, expressed: used.has(n.id)}); sourceMap.set(n.id, n.id);});
    }
    const actualNodes = new Map([...inputs, ...data.nodes].map(n => [n.id, n]));
    const buckets = new Map();
    for (const link of links) {
      const source = sourceMap.get(link.source) ?? link.source;
      // Separate opposite signs: a positive and a negative edge must never cancel in a bundle.
      const key = `${source}:${link.target}:${link.weight >= 0}:${link.enabled}:${link.expressed}`;
      const node = actualNodes.get(link.source);
      const signal = link.delayed ? node?.previous_value ?? 0 : node?.value ?? 0;
      if (!buckets.has(key)) buckets.set(key, {...link, source, count: 0, weightSum: 0, signalSum: 0, minWeight: Infinity, maxWeight: -Infinity});
      const bundle = buckets.get(key);
      bundle.count++;
      bundle.weightSum += link.weight;
      bundle.signalSum += Math.abs(signal * link.weight);
      bundle.minWeight = Math.min(bundle.minWeight, link.weight);
      bundle.maxWeight = Math.max(bundle.maxWeight, link.weight);
    }
    const edges = [...buckets.values()].map(e => ({...e, weight: e.weightSum / e.count, signal: e.signalSum / e.count}));
    return {nodes, edges, mode, inputCount: inputs.length,
      connectedInputs: new Set(data.connections.filter(e => e.enabled && e.expressed && e.source < 0).map(e => e.source)).size,
      hiddenCount: data.nodes.filter(n => n.kind === 'hidden').length,
      enabledCount: data.connections.filter(e => e.enabled).length,
      delayedCount: data.connections.filter(e => e.enabled && e.expressed && e.delayed).length};
  }

  function layoutGraph(graph) {
    const width = 1100, height = 660;
    const left = graph.nodes.filter(n => n.kind === 'input' || n.kind === 'group').sort((a,b) => a.group-b.group || (a.index??0)-(b.index??0));
    if (graph.mode === 'grouped') {
      left.forEach((n,i) => Object.assign(n, {x:245, y:72+i*39, radius:n.kind === 'group'?13:8}));
    } else {
      const columns = Math.max(1, Math.ceil(left.length / 26));
      const rows = Math.ceil(left.length / columns);
      left.forEach((n,i) => Object.assign(n, {x:48+(i%columns)*(columns === 1?0:310/(columns-1)), y:78+Math.floor(i/columns)*(rows<=1?0:520/(rows-1)), radius:4.6}));
    }
    const hidden = graph.nodes.filter(n => n.kind === 'hidden').sort((a,b)=>a.id-b.id);
    const ids = new Set(hidden.map(n=>n.id));
    const adjacent = new Map(hidden.map(n=>[n.id, []]));
    graph.edges.filter(e=>e.enabled && e.expressed && ids.has(e.source) && ids.has(e.target)).forEach(e=>adjacent.get(e.source).push(e.target));
    // Condense recurrent cycles before choosing columns; no fictitious feed-forward layers.
    const index = new Map(), low = new Map(), stack=[], onStack=new Set(), components=[];
    let cursor=0;
    function visit(id) {
      index.set(id,cursor);low.set(id,cursor++);stack.push(id);onStack.add(id);
      for (const next of adjacent.get(id)) {
        if (!index.has(next)) {visit(next);low.set(id,Math.min(low.get(id),low.get(next)));}
        else if(onStack.has(next)) low.set(id,Math.min(low.get(id),index.get(next)));
      }
      if(low.get(id)===index.get(id)) {const component=[];let current;do{current=stack.pop();onStack.delete(current);component.push(current);}while(current!==id);components.push(component);}
    }
    hidden.forEach(n=>{if(!index.has(n.id))visit(n.id);});
    const componentOf = new Map();components.forEach((values,i)=>values.forEach(id=>componentOf.set(id,i)));
    const depth=components.map(()=>0);
    for(let round=0;round<components.length;round++) {
      let changed=false;
      for(const [id, targets] of adjacent)for(const target of targets){const a=componentOf.get(id),b=componentOf.get(target);if(a!==b&&depth[b]<depth[a]+1){depth[b]=depth[a]+1;changed=true;}}
      if(!changed)break;
    }
    const maxDepth=Math.max(0,...depth);
    for(let layer=0;layer<=maxDepth;layer++) {
      const members=hidden.filter(n=>depth[componentOf.get(n.id)]===layer);
      members.forEach((n,i)=>Object.assign(n,{x:maxDepth?520+layer/maxDepth*275:660,y:120+(i+1)/(members.length+1)*430,radius:20}));
    }
    graph.nodes.filter(n=>n.kind==='output').sort((a,b)=>a.id-b.id).forEach((n,i)=>Object.assign(n,{x:984,y:245+i*210,radius:30}));
    return {...graph,width,height};
  }

  class NetworkView {
    constructor(panel, onSelection) {
      this.panel=panel;this.canvas=panel.querySelector('canvas');this.tooltip=panel.querySelector('.network-tooltip');
      this.context=this.canvas.getContext('2d');this.mode=panel.querySelector('#network-mode');
      this.disabled=panel.querySelector('#network-disabled');this.selector=panel.querySelector('#network-worm');
      this.scale=1;this.tx=0;this.ty=0;this.data=null;this.graph=null;this.hover=null;this.fitted=false;
      this.selector.onchange=()=>onSelection(Number(this.selector.value));
      this.mode.onchange=()=>{this.fitted=false;this.rebuild();};this.disabled.onchange=()=>this.rebuild();
      panel.querySelector('#network-fit').onclick=()=>this.fit();
      panel.querySelector('#network-expand').onclick=()=>{panel.classList.toggle('network-expanded');panel.querySelector('#network-expand').textContent=panel.classList.contains('network-expanded')?'Réduire':'Agrandir';this.fit();};
      window.addEventListener('keydown',event=>{if(event.key==='Escape'&&panel.classList.contains('network-expanded')){panel.classList.remove('network-expanded');panel.querySelector('#network-expand').textContent='Agrandir';this.fit();}});
      new ResizeObserver(()=>{this.resize();this.fit();}).observe(this.canvas);
      this.canvas.addEventListener('wheel',event=>{
        event.preventDefault();const rect=this.canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
        const old=this.scale;this.scale=Math.max(.2,Math.min(5,this.scale*Math.exp(-event.deltaY*.0015)));
        this.tx=x-(x-this.tx)*this.scale/old;this.ty=y-(y-this.ty)*this.scale/old;this.draw();
      },{passive:false});
      this.canvas.addEventListener('pointerdown',event=>{this.drag={x:event.clientX,y:event.clientY};this.canvas.setPointerCapture(event.pointerId);});
      this.canvas.addEventListener('pointerup',()=>{this.drag=null;});
      this.canvas.addEventListener('pointercancel',()=>{this.drag=null;});
      this.canvas.addEventListener('pointerleave',()=>{if(!this.drag){this.hover=null;this.tooltip.hidden=true;this.draw();}});
      this.canvas.addEventListener('pointermove',event=>this.pointer(event));
      this.resize();
    }
    resize(){const rect=this.canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;this.canvas.width=Math.round(rect.width*d);this.canvas.height=Math.round(rect.height*d);this.width=rect.width;this.height=rect.height;this.dpr=d;}
    fit(){if(!this.graph)return;this.scale=Math.min(this.width/this.graph.width,this.height/this.graph.height)*.98;this.tx=(this.width-this.graph.width*this.scale)/2;this.ty=(this.height-this.graph.height*this.scale)/2;this.fitted=true;this.draw();}
    update(snapshot,state){
      if(!snapshot)return;
      const selected=this.selector.value;
      const referenceGame=snapshot.worms.some(w=>w.controller==='reference');
      const signature=snapshot.worms.map(w=>`${w.id}:${w.alive}:${w.controller}`).join(',');
      if(this.wormSignature!==signature){
        this.wormSignature=signature;this.selector.replaceChildren();
        this.selector.add(new Option(referenceGame?'Auto : réseau évalué':'Auto : plus gros vivant','-1'));
        snapshot.worms.forEach(w=>{const option=new Option(`Ver ${w.id+1} · ${w.controller==='reference'?'adversaire fixe':w.alive?'vivant':'mort'}`,String(w.id));option.disabled=w.controller==='reference';this.selector.add(option);});
        this.selector.value=[...this.selector.options].some(o=>o.value===selected&&!o.disabled)?selected:'-1';
      }
      const data=snapshot.network;
      if(!data){this.panel.querySelector('#network-status').textContent='En attente du premier état du réseau…';return;}
      this.data=data;this.phase=state.status?.phase;
      const live=state.active && this.phase==='training' && data.alive && data.generation===state.status.generation;
      this.panel.querySelector('#network-status').textContent=`${live?'En direct':this.phase==='paused'?'En pause':'Instantané'} · arène ${data.arena+1} · ver ${data.worm+1} · génome ${data.genome_id}${data.species_id!=null?` · espèce E${data.species_id}`:''} · génération ${data.generation+1}${data.alive?'':' · mort, sorties non appliquées'}`;
      const boost=data.nodes.find(n=>n.kind==='output'&&n.label==='Boost')?.value??0;
      const direction=data.nodes.find(n=>n.kind==='output'&&n.label==='Direction')?.value??0;
      this.panel.querySelector('#network-boost').textContent=`Boost ${boost>=.5?'ON':'OFF'} · ${boost.toFixed(3)}`;
      this.panel.querySelector('#network-direction').textContent=`Direction ${direction.toFixed(3)} · ${((direction%1)*360).toFixed(1)}°`;
      this.panel.querySelector('#network-time').textContent=`Mesure à ${data.observation_time.toFixed(1)} s · mise à jour ≈ 1/s`;
      this.rebuild();
    }
    rebuild(){if(!this.data)return;this.graph=layoutGraph(buildGraph(this.data,this.mode.value,this.disabled.checked));
      this.panel.querySelector('#network-counts').textContent=`${this.graph.connectedInputs} / ${this.graph.inputCount} entrées reliées · ${this.graph.hiddenCount} neurones cachés · 2 sorties · ${this.graph.enabledCount} connexions activées · ${this.graph.delayedCount} liens de mémoire`;
      if(!this.fitted)this.fit();else this.draw();
    }
    pointer(event){
      if(this.drag){this.tx+=event.clientX-this.drag.x;this.ty+=event.clientY-this.drag.y;this.drag={x:event.clientX,y:event.clientY};this.tooltip.hidden=true;this.draw();return;}
      if(!this.graph)return;const rect=this.canvas.getBoundingClientRect();const x=(event.clientX-rect.left-this.tx)/this.scale,y=(event.clientY-rect.top-this.ty)/this.scale;
      const hit=this.graph.nodes.find(n=>Math.hypot(n.x-x,n.y-y)<n.radius+5/this.scale);this.hover=hit?.id??null;
      this.tooltip.hidden=!hit;
      if(hit){let text=`${hit.label}\nValeur : ${hit.value.toFixed(4)}`;
        if(hit.kind==='group')text+=` (moyenne absolue)\n${hit.connected} entrées reliées sur ${hit.count}\nMaximum : ${hit.peak.toFixed(4)}`;
        else if(hit.kind!=='input')text+=`\nValeur précédente : ${hit.previous_value.toFixed(4)}\nBiais : ${hit.bias.toFixed(4)} · réponse : ${hit.response.toFixed(3)}\n${hit.expressed?'Neurone exécuté':'Neurone non exécuté'}`;
        const incoming=this.data.connections.filter(e=>e.target===hit.id&&e.enabled);if(incoming.length)text+=`\n${incoming.length} connexions entrantes`;
        this.tooltip.textContent=text;
        this.tooltip.style.left=Math.min(event.clientX-rect.left+14,this.width-265)+'px';this.tooltip.style.top=Math.max(4,Math.min(event.clientY-rect.top+12,this.height-125))+'px';}
      this.draw();
    }
    draw(){const ctx=this.context;if(!ctx)return;ctx.setTransform(this.dpr,0,0,this.dpr,0,0);ctx.clearRect(0,0,this.width,this.height);
      if(!this.graph){ctx.fillStyle='#8194a3';ctx.font='14px system-ui';ctx.fillText('Le réseau apparaîtra avec la simulation.',24,48);return;}
      ctx.save();ctx.translate(this.tx,this.ty);ctx.scale(this.scale,this.scale);
      ctx.font='600 13px system-ui';ctx.fillStyle='#a0b4c3';ctx.fillText(this.graph.mode==='grouped'?'ENTRÉES REGROUPÉES':'NEURONES D’ENTRÉE',32,30);ctx.fillText('NEURONES CACHÉS',490,30);ctx.fillText('SORTIES',954,30);
      const byId=new Map(this.graph.nodes.map(n=>[n.id,n]));
      for(const edge of this.graph.edges){const a=byId.get(edge.source),b=byId.get(edge.target);if(!a||!b)continue;
        const selected=this.hover===a.id||this.hover===b.id;const muted=this.hover!==null&&!selected;
        ctx.globalAlpha=muted?.07:(!edge.enabled||!edge.expressed)?.13:Math.min(.8,.23+edge.signal*.5);
        ctx.strokeStyle=edge.weight>=0?'#62dcb3':'#f492a8';ctx.lineWidth=(selected?2:1)*Math.min(4,.65+Math.sqrt(Math.abs(edge.weight))+.15*Math.sqrt(edge.count));
        ctx.setLineDash(!edge.enabled?[3,6]:edge.delayed?[7,5]:[]);ctx.beginPath();
        let endX=b.x-b.radius,endY=b.y;
        if(a.id===b.id){ctx.moveTo(a.x-a.radius*.7,a.y-a.radius*.7);ctx.bezierCurveTo(a.x-70,a.y-95,a.x+70,a.y-95,a.x+a.radius*.7,a.y-a.radius*.7);}
        else if(a.x>=b.x){ctx.moveTo(a.x,a.y-a.radius);ctx.bezierCurveTo(a.x,a.y-100,b.x,b.y-100,b.x,b.y-b.radius);endX=b.x;endY=b.y-b.radius;}
        else{ctx.moveTo(a.x+a.radius,a.y);const mid=(a.x+b.x)/2;ctx.bezierCurveTo(mid,a.y,mid,b.y,endX,endY);}
        ctx.stroke();ctx.setLineDash([]);
        if(a.x<b.x){ctx.beginPath();ctx.moveTo(endX,endY);ctx.lineTo(endX-6,endY-3);ctx.lineTo(endX-6,endY+3);ctx.closePath();ctx.fillStyle=ctx.strokeStyle;ctx.fill();}
      }
      ctx.globalAlpha=1;
      if(!this.graph.hiddenCount){ctx.font='14px system-ui';ctx.fillStyle='#708795';ctx.fillText('Aucun neurone caché dans ce génome',470,610);}
      for(const node of this.graph.nodes){const selected=node.id===this.hover;const level=Math.min(1,Math.abs(node.value));
        ctx.beginPath();ctx.arc(node.x,node.y,node.radius+3,0,Math.PI*2);ctx.fillStyle='#0c1620';ctx.fill();
        ctx.beginPath();ctx.arc(node.x,node.y,node.radius,0,Math.PI*2);ctx.fillStyle=node.expressed?`rgba(${node.value<0?'238,145,166':'99,226,183'},${.15+.8*level})`:'#202d38';ctx.fill();
        ctx.strokeStyle=selected?'#ffffff':node.kind==='output'?'#e8d997':node.expressed?'#5aaf96':'#405563';ctx.lineWidth=selected?3:1.3;ctx.stroke();
        if(node.kind==='group'||(this.graph.mode==='grouped'&&node.kind==='input')){ctx.font='13px system-ui';ctx.fillStyle=node.expressed?'#d0e1ea':'#738b9a';ctx.fillText(node.label,26,node.y+4);if(node.kind==='group'){ctx.font='11px system-ui';ctx.fillStyle='#8099a9';ctx.fillText('×87',267,node.y+4);}}
        else if(node.kind!=='input'){ctx.textAlign='center';ctx.font='600 13px system-ui';ctx.fillStyle='#e8f4f7';ctx.fillText(node.label,node.x,node.y+node.radius+22);ctx.font='11px ui-monospace,monospace';ctx.fillText(node.value.toFixed(2),node.x,node.y+4);ctx.textAlign='left';}
      }
      ctx.restore();
    }
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={buildGraph,layoutGraph};
  else root.NetworkView=NetworkView;
})(typeof window==='undefined'?globalThis:window);
