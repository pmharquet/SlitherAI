# Initial NEAT training audit

**Scope:** read-only audit of `runs/20260924-163112-852097`, using the last fully evaluated population (checkpoint 3, generation 3) and its saved episode scores. At the latest status read, generation 4 was still being evaluated. No run, checkpoint, or production setting was changed. Counts below are CPU replays / allocation calculations against the saved population; they are diagnostics, not new training results.

## Findings for the next decision

### 1. Keep speciation unchanged through common validation G5

The initial 8→30 species increase is chiefly explained by structural innovations crossing the compatibility boundary, not evidence that the target range should be forced. Replaying generation 0 reproduction from checkpoint 0 and saved episode scores reproduced all 256 genomes in checkpoint 1. In those 240 offspring, 31 gained a hidden node. Of 22 new species founders, 21 had a new-node mutation and one was connection-only. Hidden-node founders had node-distance contributions 0.347–0.412 and connection contributions 1.69–1.80 (total 2.04–2.16), above the then threshold 1.8609; the connection-only founder's total was 1.889. Across the next population, the median nearest retained old representative distance was 0.049 node + 1.714 connection = 1.765, and 22 genomes were at or above threshold before new founders were created. This is consistent with NEAT's intended protection of topological innovation through speciation; it does not show that every resulting niche is useful yet. [NEAT paper](https://direct.mit.edu/evco/article/10/2/99/1123/Evolving-Neural-Networks-through-Augmenting)

Generation 0 species had sizes 13–52 (median 36). With `survival_threshold=0.3`, the within-species parent cutoffs were `[11,16,12,11,12,8,6,4]`: 80/256 genomes entered the parent pools. Elitism retained 16 genomes verbatim; the other 240 were offspring. The replay selected 80 distinct parents from those pools and produced 22 same-parent child pairs. Parent-pool restriction and per-species mating mean that species count does not directly equal the number of active lineages or the breadth of selected parents.

`AdaptiveSpeciesSet` moves the compatibility threshold by only 0.08 after each generation's speciation when the count lies outside 8–12. The count was 8→30→36→38 while threshold advanced 1.8609→1.9409→2.0209; its current implementation retains each prior species by choosing a nearest current representative and never merges existing species directly. A larger step might slow future births, but it could also stop protecting the exact innovations that caused this burst. Species count by itself is therefore not a sound reason to tune the threshold. Do not force 8–12 or add a merge rule on these four generations of evidence.

### 2. Reject `min_species_size=2` as a current intervention

Recomputed `DefaultReproduction` allocations for checkpoint 3 / generation 3 with the same adjusted fitness, `elitism=2`, and each minimum set to 4 or 2. Both settings yielded 256 total slots: 74 elite copies and 182 mutant offspring; no species landed at the effective floor and no niche was left with only its two elites. Top-four/top-eight spawn totals were 32/67 in both cases; effective species by offspring share was 35.73 versus 35.66, largest share 4.3% in both. The library uses the effective minimum `max(min_species_size, elitism)` and smooths then exactly adjusts spawn counts, which explains why lowering the configured minimum had no allocation effect here. [NEAT-Python reproduction source](https://neat-python.readthedocs.io/en/latest/_modules/reproduction.html) [configuration reference](https://neat-python.readthedocs.io/en/latest/config_file.html)

Conclusion: do not branch or change this parameter for the current population. In a later population, min 2 could let a weak species persist as two unchanged elites with no mutant descendants, so any future comparison must report elite-only niches and their fitness as well as slots freed. Re-run the allocation comparison only if a species actually receives the minimum or parent/offspring concentration becomes a measured problem.

### 3. Learning indicators are promising but not validation evidence

Across generations 0–3, mean selection fitness improved −17.69→−4.94→−4.10→−2.36; fixed-anchor fitness improved −17.47→−5.42→−4.40→−2.07; mean food gain rose 7.40→9.42→10.78→11.39, and champion best rose 10.47→13.46→18.74→22.77. At the same time, aggregate focal alive fraction was only 0.31%, 0.31%, 0.31%, and 0.78%; border deaths remained 71–80%. Those trends support continued observation, not a claim of generalization or a reward change after four generations.

Selection score uses four fixed anchor episodes at 80% weight plus one rotating episode at 20%; the anchor aggregate is half mean and half median. This gives useful common comparisons but also makes the anchor portion reusable for selection overfitting. Keep a separate final holdout, and compare challengers on the same predeclared maps, seeds, and opponents. The generation-0 fixed validation measured the learned champion at 9.45 ±2.53 SE and the heuristic at 43.88 ±4.69 SE; the heuristic is also a training opponent, and those numbers are not directly a learned-agent win comparison. G5's common validation is the next decision point.

## Replay caveat

Checkpoint 0→1 reproduction replay was exact (256/256 genomes). Replays for 1→2 and 2→3 each matched 223/256 genomes; all remaining differences were structural node/connection key sets, while the final random state and innovation tracker matched their checkpoint targets. This is unresolved. Do not attribute it to Python hash order: the current investigation has not yet proved equivalence with the full `Population.run` reporter/checkpoint sequence, pickle restore inputs, and fresh-process `PYTHONHASHSEED` variants. It blocks exact-resume claims for those transitions, not the first-transition distance evidence or the allocation calculation. A separate bounded reproducer should locate the first divergence before using these replays for deterministic A/B claims.

## Decision and reproducible follow-up

**Production decision now:** preserve the active baseline through common G5 validation; no species/reward intervention is supported by this audit. Keep topological mutation and speciation protection intact. The min-size-2 proposal is rejected for lack of an allocation effect at G3.

**Next experiment gate:** after G5, only if the fixed-seed validation and the recorded species-level evidence point to a concrete failure mode, select one speciation/reproduction parameter and fork from the same compatible checkpoint. Compare on identical anchor scenarios and budget; track validation score per episode, food, survival, border death, topology changes, parent-pool coverage, species sizes, spawn/elite counts, and anchor-versus-rotating scores. Do not use the final holdout to tune. Before any exact-resume A/B, complete the fresh-process/full-production-path replay reproducer and record its first divergent child.

## Parameters and interactions to watch

| Control | Current value | Interaction/evidence |
|---|---:|---|
| Compatibility threshold / step / target | 1.86 initially; step .08; target 8–12 | Count feedback is one-generation delayed relative to births and species are not directly merged; new-node distance often tips genomes across the threshold. |
| Population / minimum species size / elitism | 256 / 4 / 2 | Effective minimum is 4; G3 min=2 gave identical allocation. A smaller floor can create elite-only niches if an allocation reaches 2. |
| Survival threshold | .30 | Controls per-species parent pool separately from spawn allocation; G0 supplied 80 parents across eight niches. |
| Structural mutation | node-add .15, connection-add .50, node-delete .02, connection-delete .05; multi-structural mutations allowed | 31 node-add offspring in G0→G1 replay; hidden-node changes were common among new-species founders. Keep innovation protection during diagnosis. |
| Stagnation | 30 generations; 4 species protected; max 1 removal/generation | Far too early to infer cleanup behavior at G0–G3; it will not quickly collapse excess species. |
| Evaluation | 4 fixed + 1 rotating; 80/20 | Fixed-anchor optimization can diverge from performance on independent maps; preserve paired scenarios and holdout. |

Current NEAT-Python is pinned at 1.1.0. Newer 2.1 documentation mentions `target_num_species`, but that is not a current parameter and is not a drop-in fix. Generic Reddit anecdotes did not provide causal evidence relevant to this measured burst; primary NEAT and installed-library behavior are more informative here.
