const test = require('node:test');
const assert = require('node:assert/strict');
const { buildGraph, layoutGraph } = require('../web/network-view.js');

function fixture() {
  return {
    schema:{ray_channels:['range','head','body','own','border','food'],angles_degrees:Array.from({length:87},(_,i)=>i-43)},
    input_keys:Array.from({length:530},(_,i)=>-i-1), inputs:Array.from({length:530},()=>.25),
    nodes:[{id:0,kind:'output',label:'Boost',value:.4,previous_value:.1,expressed:true},
      {id:1,kind:'output',label:'Direction',value:.2,previous_value:.3,expressed:true},
      {id:2,kind:'hidden',label:'N2',value:.8,previous_value:.9,expressed:true},
      {id:3,kind:'hidden',label:'N3',value:.7,previous_value:.5,expressed:true}],
    connections:[
      {source:-1,target:2,weight:1,enabled:true,expressed:true,delayed:false},
      {source:-7,target:2,weight:-2,enabled:true,expressed:true,delayed:false},
      {source:-13,target:2,weight:3,enabled:false,expressed:false,delayed:false},
      {source:2,target:3,weight:1,enabled:true,expressed:true,delayed:true},
      {source:3,target:2,weight:1,enabled:true,expressed:true,delayed:true},
      {source:2,target:0,weight:1,enabled:true,expressed:true,delayed:true},
      {source:1,target:1,weight:.3,enabled:true,expressed:true,delayed:true}]
  };
}

test('grouped inputs preserve positive/negative bundles and recurrent source activity',()=>{
  const graph=buildGraph(fixture());
  const group=graph.nodes.find(n=>n.id==='group-0');
  assert.equal(group.count,87);
  assert.equal(group.connected,2);
  const incoming=graph.edges.filter(e=>e.source==='group-0');
  assert.equal(incoming.length,2);
  assert.deepEqual(incoming.map(e=>e.weight).sort(),[-2,1]);
  assert.equal(graph.edges.find(e=>e.source===2&&e.target===3).signal,.9);
});

test('detailed inputs and disabled connections retain exact identities',()=>{
  const graph=buildGraph(fixture(),'connected',true);
  assert.equal(graph.nodes.filter(n=>n.kind==='input').length,3);
  assert.equal(graph.nodes.find(n=>n.id===-13).expressed,false);
  assert.equal(buildGraph(fixture(),'all').nodes.filter(n=>n.kind==='input').length,530);
  assert.equal(graph.connectedInputs,2);
});

test('recurrent cycles and disconnected inputs get finite stable positions',()=>{
  for(const mode of ['grouped','connected','all']) {
    const graph=layoutGraph(buildGraph(fixture(),mode));
    assert.ok(graph.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));
    assert.equal(graph.nodes.find(n=>n.id===2).x,graph.nodes.find(n=>n.id===3).x);
    assert.equal(graph.nodes.find(n=>n.id===0).x,984);
  }
});
