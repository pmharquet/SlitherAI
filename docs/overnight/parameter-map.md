# Simulation and NEAT parameter map

Read-only inventory for the active run `runs/20260924-163112-852097`. Values below come from its saved `settings.json`, `status.json`, and `configs/neat.ini`; changing a setting on resume is rejected or changes the experiment's meaning. The run-statistics statements below refer to the audit snapshot generated at 2026-09-24 16:57 UTC: G0–G2 were complete and G3 was in progress ([audit report](../../runs/20260924-163112-852097/analysis/overnight/audit.md)). This complements the [audit-tool notes](audit-tool.md) and [performance review](performance-review.md), which contain the report format and bounded `sensor_chunk` benchmark.

Evidence labels describe local evidence only:

- **Tested locally** means a unit or regression test exercises that mechanism. It does not establish agreement with the original Slither game.
- **Implemented, calibration unknown** means the code behavior is defined, but game telemetry or an external reference has not validated the chosen value.
- **Measured, narrow scope** means a benchmark or parity probe exists, with its limits stated.

## Run budget and evaluation

| Setting | Active value | Effect and interaction | Evidence |
|---|---:|---|---|
| Population | 256 genomes | Each genome is scored over five episodes per generation; the best genome's score is not representative of the population mean. | Implemented; resume/evolution determinism is tested locally. |
| Generations | 47 more from checkpoint 3; absolute target 50 | The target is the saved status target. A generation index in files starts at zero. | Run metadata. |
| Episode duration | 90 simulated seconds | Decision interval is `dt=0.1`, or 900 action opportunities before early death. | Implemented; full-game duration equivalence unknown. |
| Evaluation batch | Up to 64 maps, 16 worms/map | The 256 candidates are evaluated in four batches for each of five scenarios: 1,280 candidate episodes per generation. `maps` also controls tensor size and GPU memory. | Candidate order/batch parity tested on small CPU/CUDA cases. |
| Seed | 1 | Fixes initial population, anchor scenario seeds, and training randomness on a fresh run; actions still cause trajectories to diverge. | Repeat/resume behavior tested locally. |
| Device | CUDA | Tensor simulation and network activation use the RTX 4060 Laptop GPU; NEAT mutation, species, and reproduction run on CPU. | CPU/CUDA network and scenario parity tests exist; full-run fidelity is not a device property. |
| Validation cadence | First evaluated generation, then every 5 generations (`validation_every=5`) | In zero-based files: generations 0, 4, 9, 14, … . The current saved record is only generation 0. | Implemented; validation is a model-selection set, not an independent test. |

## `SimConfig` values

| Parameter(s) | Active value | Causal effect and important interaction | Evidence |
|---|---:|---|---|
| `maps`, `worms` | 64, 16 | Independent arenas are the batch dimension; each candidate occupies one arena against 15 reference-policy opponents. More maps increase throughput opportunity and GPU memory; more worms increase contact pressure. | Map independence and small-batch parity tested locally. Opponent density is an approximation. |
| `foods`, `preys` | 1,024, 8 | Ambient food is replenished after eating. The first eight food entries move as prey; prey flee within 150 units. More food or fewer worms changes food per competitor and survival pressure. | Eating and food conservation are tested; density is not calibrated to the original game. |
| `body_points`, `body_spacing` | 96, 9 units | Fixed polyline point count; spacing is `max(9, body_length/(96−1))`, so long worms get coarser body samples. Affects collision/sensor detail and cost. | Long-body culling parity and collision geometry tests exist; external body sampling is unknown. |
| `arena_radius`, `arena_variation` | 2,400 units, ±20% | Each map samples a uniform radius in `[1,920, 2,880]`. Smaller arenas and 16 worms increase edge/contact opportunities. Ambient food spans 98% of radius. The docs cite an exported in-game radius estimate near 31,899 units; this local arena is deliberately much smaller. | Implemented; arena size and encounter density are explicitly approximate. |
| `view_half_width`, `view_half_height` | 600, 500 units | Rectangular visible range, multiplied by `sqrt(worm_radius/14.5)`. Growth expands view while slowing turns, so size changes both sensing and maneuverability. | Implemented; viewport scaling is not externally calibrated. |
| `dt`, `substeps` | 0.1 s, 3 | One policy action per 0.1 s; movement, turning, body follow, age, and collision checks update three times within that interval. More substeps cost more simulation but reduce discrete movement per collision check. | Movement, turn bounds, and collisions tested locally; step size fidelity unknown. |
| `base_speed`, `boost_speed`, `turn_rate` | 115.8, 210 units/s, 2.8 rad/s | Actual speed is multiplied by `1 + 0.1×(radius/14.5−1)`. Turn limit per substep is `turn_rate×sqrt(14.5/radius)×dt/substeps`: larger worms move faster under this formula but turn more slowly. | Turn and boost behavior tested; speed conversion and size response are uncalibrated. |
| `raw_speed_scale` | 20 | Live speed input is `speed/(20×12)`, capped at 1; at defaults, base/boost speeds encode as about 0.48/0.88 before size scaling. Corresponds to the documented conversion from `speedRaw` to world units. | Code and importer agree at the documented scale; source-game speed remains an estimate. |
| `initial_mass`, `min_mass` | 35, 10 | Spawn mass is randomized to 80–120% of 35. Mass determines radius, segments, length, food attraction context, and the boost eligibility floor. | Implemented; mass and growth curves are approximate. |
| `boost_cost` | 3 mass/s | Each 0.1 s action spends 0.3 mass while boost is active, if mass exceeds `min_mass + cost×dt`. That mass is dropped behind the worm and can be eaten again; reward uses net food minus boost spend. | Boost cost and deposited food are tested; source-game cost is uncalibrated. |
| `sensor_chunk` | 4 worms per sensing block | Changes the raycast work-block size, not the observation values. Larger chunks lower launch/loop overhead at a small memory cost. | **Measured, narrow scope:** 4/8/16 gave bitwise-identical observations; spawn-state CUDA tick medians were 55.54/36.82/28.00 ms for observation+physics. Full grown-body generation throughput remains unmeasured. |
| `reward_version` | `growth-v2` | Selects the active objective. Old runs without a version are read as `legacy-v1`; their fitness histories cannot be mixed with v2. | Formula and version guard tested locally. |

## Other fixed simulation constants

These are implemented in `slitherai/sim.py`, not exposed as `SimConfig` fields. They can still affect what NEAT learns.

| Mechanic and constants | Effect / interaction | Evidence |
|---|---|---|
| Eight food hotspots; 30% of random placements use a hotspot with ±140-unit square jitter; other food is uniform over a disk at 98% arena radius. | Creates repeatable food-rich regions within each map; respawn maintains ambient food count. The hotspot geometry changes routes and local congestion. | Implemented; hotspot distribution is a design choice, not measured against game exports. |
| Ambient food mass `0.5 + U[0,2.5]`; displayed size `clamp(3×mass, 2, 20)`. | Food gain changes mass, radius, length, speed multiplier, view, and turn limit. | Food acquisition is tested; exact food economy unknown. |
| Worm radius `14.5×(1+mass/300)^0.4`, capped at 80; segments `2+mass/15`; length `20×segments`. | Couples food growth to body collision size, turn rate, view, and body point spacing. | Implemented; curves are approximate. |
| Enemy head sensor radius `r_self+r_enemy+3`; enemy body sensor radius also adds 3. Own-body sensing omits the nearest `3×r_self`; own-body collision is safe. | Sensor clearance and lethal collision clearance differ; the self-body channel describes geometry that is not itself fatal. | Own-body safety and enemy collision are tested; sensor offset fidelity unknown. |
| Collision rule: head against enemy-body capsules or `|head|+radius >= arena`; checked each physics substep. | A head can die at either edge or body. When both conditions occur in a substep, the event is classified as a collision. | Local kill cases tested; original game collision order/radii unknown. |
| Death drops 70% of mass across active body points. Boost drops 0.3 mass/s in a per-worm circular buffer of 96 slots; oldest deposits are overwritten. | Death food and boost recycling alter food density and whether a boost route is profitable. | Food/boost mechanics tested in parts; drop fraction and reservoir policy are local assumptions. |
| Prey speed 75 units/s; random heading drift ±0.25 rad/action; flee within 150 units; turn inward beyond 95% of arena radius. | Moving food changes route timing and opponent interactions. | Implemented; prey model is not calibrated. |

## Observation, normalization, and action

The saved contract is `slither-neat-530-v1`: 87 rays × 6 channels + 8 globals = 530 inputs. Rays cover `−166…−94°` at 8° steps, `−88…−48°` at 4°, `−44…44°` at 2°, then mirrored positive ranges; the rear blind cone is 28° wide. Ray order, channels, and absolute-heading outputs are shared with the data converter ([`schema.py`](../../slitherai/schema.py), [`sim.py`](../../slitherai/sim.py)).

| Input/action | Live rule | Effect / evidence |
|---|---|---|
| Visible range | `range/(range+1000)` where range is the ray exit from the zoomed rectangular view. | Bounded scale preserves a distance cue without unbounded values. Local calculation is implemented; original viewport scale is unknown. |
| Enemy head/body, own body, border | For a hit within ray range: `1/(1+distance/100)`; otherwise 0. Each ray takes nearest hit of each family. | Larger values mean nearer. Tests cover analytic ray-circle/capsule geometry and collision cases, but not a comprehensive channel-by-channel external lidar match. |
| Food/prey | Per angular beam, maximum of `min(food_size/20,1)/(1+entry_distance/100)`; food behind the 170° cutoff is ignored. | Large nearby food weighs more. The food channel is approximate: foods do not occlude one another, and beam projection is a local model. |
| Global speed/radius/length | `speed/(raw_speed_scale×12)`, `radius/80`, `segments/400`, each capped at 1. | Growth changes three global cues at once. |
| Global heading/turn error/previous boost | Sine and cosine pairs for absolute heading and requested-minus-current turn; previous boost bit. | Circular encodings avoid the 0/1 direction seam; neural state adds a separate recurrent memory. |
| Boost output | Sigmoid output 0 enables nothing; output ≥0.5 requests boost, then mass gate applies. | Hard threshold creates a discontinuity around 0.5. Threshold and resource floor are implemented/tested locally. |
| Direction output | Sigmoid output maps to absolute turns `[0,1)`, wrapped modulo 1; 0 and 1 are east. `step()` moves only by the size-limited turn rate. | Output is an absolute target, not a relative steering angle. Wrap and turn bounds are tested. |

The importer maps external `speedRaw/12`, body radius `/80`, and segment count `/400`; live speed uses `/raw_speed_scale/12` with `raw_speed_scale=20`. Other values use the 100-unit proximity and size-20 food scales. Agreement at those scales does not establish that the local geometry matches the source game.

## NEAT genome and phenotype

The active configuration is `configs/neat.ini`; `load_config()` overrides the file population size with the run's `population` value. NEAT-Python 1.1.0 owns mutation, crossover, innovations, and reproduction. `BatchedNetwork` implements the phenotype on CPU/CUDA.

| NEAT parameters | Active value | Causal effect / interaction | Evidence |
|---|---|---|---|
| Task/termination | `max`; fitness threshold 1,000,000; reset on extinction `True`; fitness termination disabled | Best-score criterion is recorded, but run duration is set by requested generation count. | Implemented; the large threshold is not expected to stop these runs. |
| Topology | 530 inputs, 2 outputs, 0 hidden at initialization; feed-forward `False`; 10% `partial_direct` initial connections | Starts with sparse direct input→output links; hidden and recurrent links can appear later. No fixed hidden-node maximum is configured. | Recurrent CPU/CUDA output parity tested, including added hidden nodes. |
| Activation/aggregation | Sigmoid only; sum only; mutation rates both 0 | Activation and aggregation are fixed. Network node update is synchronous: recurrent sources read the previous tick's state. | Parity tested against NEAT-Python. CUDA evaluator rejects any non-sigmoid/non-sum genes. |
| Weights | Gaussian init mean 0, SD 0.15; clamp ±10; mutate rate 0.35, power 0.08; replace rate 0.01 | Controls initial input contribution and continuing weight adaptation. Sigmoid uses an effective factor 5, so weight/bias scale interacts strongly with saturation. | Code/config; a 16-genome initial probe found outputs 0.092–0.903, too small to establish population-wide saturation. |
| Bias | Gaussian mean 0, SD 0.1; clamp ±10; mutate rate 0.35, power 0.08; replace rate 0.02 | Shifts each node's activation independently of inputs. | Code/config; no parameter sweep. |
| Response | Mean 1, SD 0; clamp 0.1–10; mutate rate 0.1, power 0.05; replace rate 0 | Scales weighted sum before sigmoid; can amplify or damp input/recurrent signals. | Code/config; no parameter sweep. |
| Enabled genes | Default `True`; mutation 0.01; add-to-true 0.02; add-to-false 0 | Re-enables/disables links during mutation. | Code/config; not independently benchmarked. |
| Genome distance | Disjoint coefficient 1.0; weight coefficient 0.5 | Sets how topology and weights contribute to compatibility and species membership. | Speciation mechanics tested; threshold behavior is run-dependent. |
| Structural mutation | Add connection 0.50; delete connection 0.05; add node 0.15; delete node 0.02; single structural mutation `False`; `surer=default` | Several structural mutation opportunities may occur per offspring; node addition splits a link in NEAT-Python. Adds expressivity and computational cost. | Mutation/restart behavior covered indirectly; no blind sweep justified by current metrics. |

The local phenotype computes `sigmoid(clamp(5×(bias + response×sum), −60, 60))`. It has an output state plus hidden states; each tick uses current inputs and the prior tick's recurrent state. State resets at the start of each episode. Output order is boost then absolute direction. These are implementation facts, not evidence that NEAT discovers an effective policy.

## Species and reproduction

| Parameters / rule | Active value | Causal effect / interaction | Evidence |
|---|---:|---|---|
| Adaptive compatibility threshold | Start 1.9; initial calibration targets 8–12 species in up to 12 binary-search steps over `[0.3,4.0]`; later step ±0.08 | If current count is below target, threshold decreases; above target, it increases. `min_species_size=4` caps feasible species count at population/4. | Initial calibration, target response, and capacity constraint tested locally. |
| Current state | Threshold 2.0209375; 38 current species, effective diversity 34.27; target 8–12 | Snapshot is during generation 3. Counts were 8, 30, and 36 for the first three completed evaluations. In the audit summary, “tiny” means size ≤ `max(2, ceil(1% of population))`, or size ≤3 here; 4 current species contain 8 genomes. High counts can partition breeding niches and interact with the four-genome minimum; this does not prove diversity is useful or harmful. | Read from current run telemetry; “tiny” is a descriptive audit cutoff, not a NEAT control. |
| Species score/stagnation | Mean fixed-anchor score; compare medians of 5-generation windows; improvement must exceed 0.25; eligible after 30 generations; protect top 4; remove at most 1/generation | The rotating fifth episode does not reset stagnation. This separates stable-scenario progress from rotating-scenario noise. | Mean/progress/protection/removal behavior tested locally. |
| Reproduction | Elitism 2; survival threshold 0.3; min size 4 | Two top members are preserved per species subject to NEAT's allocation; parents come from a 30% survival threshold. Reproduction fitness still uses combined per-genome selection fitness; anchor score controls stagnation/protection. | Species allocation and deterministic-resume tests exist. |

## Reward and scenario protocol

| Setting | Active value | Causal effect / interaction | Evidence |
|---|---:|---|---|
| Reward (`growth-v2`) | `25×asinh((food_gain−boost_spent)/25) + 0.02×age + 2×sqrt(kills) − 12×death` | Net growth dominates; 90 seconds alive without growth gives 1.8 points. A death incurs −12; kills have diminishing returns. Boost re-eaten by the same worm cannot create net reward by itself. | Formula, legacy read behavior, and edge cases tested locally. Constants are design choices, not game truth. |
| Fixed anchor scenarios | Four of five episodes; seeds `seed+100003`, `+200003`, `+300007`, `+400009`; focal slots 0,5,10,15 modulo worm count | Same anchors recur across generations. Score is half the mean plus half the median across four anchor episodes. | Seed reuse and robust aggregation tested. |
| Rotating scenario | One episode; seed `seed+10000019+generation×1009`; focal slot `generation mod worms` | Broadens conditions but introduces generation-specific score movement. Each candidate receives identical exogenous random draws within a scenario batch; their actions can make states diverge. | Slot/batch/randomness parity tested locally. |
| Selection / stagnation | Selection `0.8×anchor_score+0.2×rotating_score`; anchor score alone feeds species progress/stagnation | The run's selection mean and `evaluation.fitness` (raw episode outcome mean) are distinct telemetry fields. | Aggregation tested; learning benefit of 20% rotating weight unknown. |
| Fixed validation | 32 maps, 90 simulated seconds, seed 938271; validation every 5 generations after the first; save best mean | Uses a separate seed from training but the same fixed validation cases across checkpoints; best-validation selection can overfit those cases. Baseline food/avoidance and circle policies are calibrated once on this setup. | Per-map standard error calculated; no independent holdout result exists. |

## Boundary sensor check — possible measurement gap

`WorldBatch.observe()` computes the border channel by solving the positive ray exit from a circle of radius `arena−worm_radius−3`, then applies the same 100-unit proximity function as other hits and masks it beyond the rectangular view range. `_collisions()` kills at `head_distance+worm_radius >= arena`. For a live head inside the circle, the ray-circle equation has the expected orientation and the 3-unit inset makes the signal slightly earlier than the collision surface; I found no sign or angle-wrap mismatch in the code.

The current high border-death rate is therefore not proof of a bad sensor formula. In the 16:57 UTC audit snapshot, 73.7% of candidate episodes were labeled border deaths versus 26.0% collision deaths, and population survival to episode end was 0.3%. Simultaneous edge-and-body deaths are assigned to collision (`border_deaths` explicitly excludes a collision), so the measured border fraction can undercount edge involvement. However, `test_self_body_is_safe_enemy_body_and_border_kill` checks the death outcome, not the numeric border input, distance-to-collision timing, all ray directions, radius changes, or the view-range cutoff. **Potential issue:** the boundary signal's consistency with the lethal geometry is untested end to end, and the death categories are exclusive despite possible co-causes. A small deterministic sensor-to-collision probe would resolve this before considering any boundary-related parameter change; keep the candidate's actions neural during the probe.

## Prioritized hypotheses from this run

1. **Check boundary observability before changing boundary-related settings.** The large edge-death share, small local arena, coarse peripheral rays, and size-limited turn rate interact. First test whether the candidate receives a useful border value before edge deaths on the same local geometry. The existing death test alone does not answer that.
2. **Wait for the next fixed validation before interpreting early progress.** Population best and mean improved through the first three completed generations, but fixed validation has only one point (`fitness=9.45`, SE `2.53`, 32 maps). The run uses that same fixed set for model selection. Keep reward, mutation, and scenario weights unchanged until another validation and the training-telemetry review are available.
3. **Treat species count as a measured discrepancy, not a reason to tune the threshold immediately.** Current count 38 is well above the 8–12 target after three evaluations; the adaptive rule changes the threshold only 0.08 per generation. Inspect distances, size allocation, and anchor fitness in the species review before considering a threshold/rate change.
4. **Keep `sensor_chunk` as a throughput experiment only.** The existing `sensor_chunk=16` observation benchmark has exact value parity and lower spawn-state tick time, but does not cover grown bodies or completed-generation throughput. It cannot be changed on this resumed run because the saved simulation config must match; test it on a new controlled run only if full-episode measurements justify the provenance cost.

## Fidelity boundary

Unit tests establish local geometry, food allocation, boost cost/turn wrapping, self-body safety, fixed scenario aggregation, recurrent CPU/CUDA parity, species adaptation/stagnation, and deterministic resume. The external game has not been validated for local arena size, food economy/hotspots, body growth/radius, speed/turn curves, sensor viewport and ray returns, prey motion, or collision order. A model that scores well here is a simulator policy until an independent source-game evaluation shows otherwise.
