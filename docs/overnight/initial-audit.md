# Initial NEAT training audit

**Run:** `runs/20260924-163112-852097`. Population, checkpoint, episode, and validation files were read only. CPU replays used saved episode scores as fitness callbacks; they did not simulate episodes. No GPU job was started by this audit. The separate temporary output directories used by the reproducer are deleted at exit.

## Current evidence and decision

At the fresh status read (19:39 Paris), training remained active on CUDA at generation 6 of 50. History through generation 5 showed best score 34.82, mean selection score 0.662, and 38 species. The latest fixed validation is generation 4 (G5): fitness 14.069 versus 9.451 at G1, food gain 25.783 versus 21.843, and survival on 13/32 versus 7/32 maps. Paired fitness difference was +4.618 (SE 3.500; approximate 95% interval −2.243 to 11.479), with 20 maps better and 12 worse. This is promising but inconclusive; retain the active reward and speciation settings.

The score history through generation 3 was improving, but is not a substitute for validation: `history.mean` (mean aggregate selection score) was −17.93, −5.31, −4.51, −2.71; the separate raw `evaluation.fitness` episode mean was −17.69, −4.94, −4.10, −2.36. Fixed-anchor fitness improved −17.47 to −2.07 over those generations, while focal alive fraction remained 0.31–0.78% and border deaths 71–80%. Keep the predeclared common validation and independent final holdout for decisions.

## Speciation burst: retain structural protection

The 8→30 species increase from generation-index 0 to checkpoint-1's population is consistent with new structural genes crossing the compatibility boundary. The full `Population.run` replay, using saved index-0 scores, reproduced all 256 genomes in checkpoint 1. Among its 240 offspring, 31 had a hidden node absent from both selected parents and 114 had a connection absent from both. There were 22 new species founders; 21 had a new node absent from both parents, one was connection-only. The new-node founders had node-distance contributions 0.347–0.412 and connection contributions 1.674–1.795 (total 2.038–2.156), above the 1.8609 threshold. The connection-only founder total was 1.88894. Among all checkpoint-1 genomes, the median nearest retained old-representative distance was 0.03960 node + 1.72463 connection = 1.77737; 27/256 were at or above threshold before newly created representatives were considered. This supports structural innovation as the main cause of the initial burst; it does not establish that every niche is useful. NEAT uses speciation to protect innovations while topology and weights evolve ([original NEAT paper](https://direct.mit.edu/evco/article/10/2/99/1123/Evolving-Neural-Networks-through-Augmenting)).

G0 species sizes were 13–52 (median 36). With `survival_threshold=0.3`, within-species parent pool cutoffs were `[11,16,12,11,12,8,6,4]`, or 80/256 genomes. The 240 children used 80 distinct parent IDs and 22 child pairs selected the same parent twice. Thus species count alone does not measure parent coverage or reproductive contribution.

The adaptive threshold began at 1.8609 and advanced by 0.08 per generation outside the configured 8–12 target. Species counts were 8→30→36→38 while thresholds reached 1.8609→1.9409→2.0209. The local species implementation retains old species IDs by assigning nearest representatives and does not directly merge prior species. Raising the threshold step or forcing the target range could erase the very innovations that produced these founders. **Do not tune speciation based only on this count.**

## Reproduction floor comparison: `min_species_size=2` rejected

At checkpoint 3 / generation 3, I ran the installed `DefaultReproduction.reproduce()` twice with the same saved fitness, `elitism=2`, and only the minimum size changed. Checkpoint 3 has 38 species, including two singletons; its realized allocation is 74 elite copies and 182 mutant children, not the configured maximum of 76 elites. Both settings spawned 256 total genomes and retained all 38 species.

With minimum 4, spawn counts were `{4:1, 5:9, 6:10, 7:9, 8:2, 9:3, 10:3, 11:1}`; one species reached the four-genome floor but still received two children. With minimum 2, counts were `{4:3, 5:5, 6:12, 7:9, 8:2, 9:3, 10:3, 11:1}`. Six species moved one slot each, while total elites and children stayed 74 and 182. Top-four/top-eight spawn was 41/76 in both; largest species share was 4.30% in both. Effective child-species count shifted slightly from 35.96 to 35.83. No niche was left with only its elites, and no slots were freed. NEAT-Python floors spawn at `max(min_species_size, elitism)` and then adjusts to population size ([v1.1.0 reproduction source](https://github.com/CodeReclaimers/neat-python/blob/v1.1.0/neat/reproduction.py)).

**Decision:** do not branch or change this parameter for the current population. Lowering it slightly redistributes slots but creates no additional mutant budget here. In a future population, min 2 could permit a weak niche to persist as two unchanged elites; only reconsider if the actual allocation shows a niche at that floor or a measured parent/offspring concentration problem.

## Checkpoint replay: structural provenance shift, current behavior unchanged

Replayed `checkpoint-1→checkpoint-2` and `checkpoint-2→checkpoint-3` in three fresh processes with `PYTHONHASHSEED` 0, 1, and 42. (NEAT checkpoint N stores the next population labeled for generation index N; UI/history names display that index one-based as G{N+1}.) All six trials produced the same result: 223/256 exact genome records, with 33 raw key-set mismatches. The production `Population.run` and reporter order were used; saved scores replaced simulation evaluation and validation callbacks were stubbed. Random state, species records, and innovation tracker counters matched their target checkpoints.

The mismatch is now explained locally. NEAT-Python 1.1.0's `DefaultGenomeConfig.__getstate__` calls `next(node_indexer)` while creating pickle state ([version-pinned source](https://github.com/CodeReclaimers/neat-python/blob/v1.1.0/neat/genome.py)). `TrainingReporter.save_genome` serializes the live config before reproduction; `AtomicCheckpointer` serializes it again at end-generation. Instrumented G1→G2 replay showed the champion artifact advance the node counter 34→35 and the checkpoint 68→69; replay without those serialization advances ended one ID behind the corresponding uninterrupted checkpoint. G2→G3 showed the same pattern at 69→70 and 103→104. The same results across hash seeds reject Python hash order as the explanation here.

All 33 mismatched genomes per transition are graph-isomorphic after hidden-node relabeling. Five-step recurrent CPU activations on deterministic inputs were bit-exact for every pair (maximum output delta 0). **This is a structural-key / future lineage reproducibility issue, not evidence of a current policy-behavior or fitness change.** Different keys can still affect later gene matching, crossover, and innovation tracking, so make serialization pure before relying on exact resume or deterministic A/B claims. The old checkpoints are historical artifacts; the fix cannot retroactively remove their offset.

## Recommended actions and reproducible evidence

1. Keep current growth-v2 reward, compatibility settings, and reproduction floor through further common validation. The G5 result is inconclusive and supports no learning-rule intervention.
2. Make checkpoint and champion serialization side-effect-free by pickling a shallow config snapshot with a copied `itertools.count`; preserve checkpoint tuple/schema compatibility. Test counter purity for both artifacts and failure paths, then compare uninterrupted against resumed CPU evolution for at least two generations with node additions forced. Review before restarting the live trainer. This fixes provenance; it is not expected to increase fitness by itself.
3. If later validation identifies a specific failure, test one production parameter at a time from the same compatible checkpoint and common scenario set. Record per-map score, food, survival, deaths, topology, parent coverage, spawn/elite counts, and anchor versus rotating performance. Keep the final holdout out of tuning.

The compact JSON files beside this report are the captured historical results, produced before the snapshot fix. Since the diagnostic imports `slitherai.train`, rerunning it at a fixed revision reproduces that revision's serialization behavior. To regenerate the pre-fix replay at commit `3be735b` in an isolated detached worktree, run from the project root:

```powershell
$historical = Join-Path $env:TEMP 'slitherai-audit-3be735b'
git worktree add --detach $historical 3be735b
$repoRoot = (Get-Location).Path
$auditOut = Join-Path $env:TEMP 'speciation-replay-historical-seed0.json'
$env:PYTHONHASHSEED='0'
.venv/Scripts/python.exe (Join-Path $historical 'docs/overnight/diagnose_speciation.py') --run (Join-Path $repoRoot 'runs/20260924-163112-852097') --transition 1 --transition 2 --no-allocation --output $auditOut | Out-Null
git worktree remove $historical
```

The replay reads `checkpoint-N`, `episodes/generation-NNNN.json`, settings, and history; generated reporters/checkpoints are written in a temporary directory. It does not run game simulations. The outputs in `speciation-diagnostic-seed0.json`, `speciation-replay-seed0.json`, `speciation-replay-seed1.json`, `speciation-replay-seed42.json`, and `reproduction-allocation-g3.json` remain the captured historical evidence. Running the helper after the fix should show the effect of pure snapshots instead. Reproduction floor behavior is based on the installed NEAT-Python 1.1.0 source; its config and mutation details are in the [version-pinned config](https://github.com/CodeReclaimers/neat-python/blob/v1.1.0/neat/config.py) and [original NEAT paper](https://direct.mit.edu/evco/article/10/2/99/1123/Evolving-Neural-Networks-through-Augmenting). Generic Reddit anecdotes did not add causal evidence for this measured behavior.

## Parameters and interactions to monitor

| Control | Current value | Measured interaction / priority |
|---|---:|---|
| Compatibility threshold / step / target | 1.86 initially / 0.08 / 8–12 | New-node distance often crossed threshold. Leave unchanged absent evidence that innovations are not useful. |
| Population / min species size / elitism | 256 / 4 / 2 | G3 min 2 moved six single slots but freed no offspring. Recompute from actual species sizes and reproduction, not maximum elitism. |
| Survival threshold | 0.30 | Limits parent pool within each species independently of spawn allocation; G0 used 80 distinct parents. |
| Structural mutation | node-add 0.15, connection-add 0.50, node-delete 0.02, connection-delete 0.05; multiple allowed | 31/240 G0 children had a new node absent from both parents. Preserve mutation and innovation protection. |
| Stagnation | 30 generations; 4 species protected; max 1 removal/generation | G0–G5 is too early to infer long-term cleanup. |
| Evaluation mix | 4 fixed + 1 rotating; 80/20, anchors half mean/half median | Reuse of anchors can overfit. Keep common validation and final independent holdout. |
| Serialization | champion and checkpoint config pickles | Live node counter was consumed twice per completed generation; highest-priority correctness fix is pure snapshotting. |
