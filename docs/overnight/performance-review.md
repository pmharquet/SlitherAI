# Performance and observation review

## Finding

`sensor_chunk=16` remains a tuning candidate, but the full policy trace did not preserve per-step state exactly, so the current evidence does not support adopting it. In `WorldBatch.observe()`, each raycast block covers `sensor_chunk * worms` observers ([sim.py:257](../../slitherai/sim.py#L257)); raising the chunk from 4 to 16 cuts that loop from 16 blocks to 4 for the default 64 maps and 16 worms. The initial-world CUDA sweep and later long-body fixture had bitwise equal observation outputs.

| `sensor_chunk` | Observe median | Observe + `WorldBatch.step()` median | Peak allocated | Observation parity |
|---:|---:|---:|---:|---|
| 4 | 43.17 ms | 55.54 ms | 132.97 MiB | reference |
| 8 | 24.77 ms | 36.82 ms | 134.23 MiB | exact |
| 16 | 15.03 ms | 28.00 ms | 135.75 MiB | exact |

Run conditions: RTX 4060 Laptop GPU, PyTorch 2.8.0+cu129, 64 maps × 16 worms, 1,024 foods, 96 body points, seed 1, 8 warmup ticks, 6 synchronized repeats, 1,024 alive worms and 12 active body points. The timed world tick includes observation and simulator physics, but no network activation or policy selection. This is a short warmed spawn-state microbenchmark; it does not cover late-game body growth, deaths, crowded sensors, or generation throughput. The peak allocation difference from chunk 4 to 16 was 2.78 MiB in this run.

The benchmark helper is [benchmark_observation.py](../../slitherai/benchmark_observation.py). Run it after reserving a training pause window with:

```powershell
.\.venv\Scripts\python.exe -m slitherai.benchmark_observation --maps 64 --worms 16 --foods 1024 --body-points 96 --seed 1 --warmup-ticks 8 --repeats 6 --chunks 4 8 16
```

An added CPU parity case uses long bodies, different map layouts, food locations, and two dead worms; chunk sizes 1, 2 and 4 produce bitwise equal observations ([test_observation_benchmark.py](../../tests/test_observation_benchmark.py)). This covers more body and culling states for correctness, but says nothing about CUDA performance in late-game conditions.

## Paired policy experiment for the next approved window

The helper supports paired episode runs against a run's saved `champion.pkl` and `settings.json`. It uses the champion in each map, the production `evaluation.play_episode()` loop and fixed `evaluation.heuristic` opponents, the validation focal-seat pattern `map_index * 7 % worms`, independent maps within each episode, and common seeds across chunk sizes. It runs uninstrumented episodes for timing, then separate traced episodes that compare per-step observation, action, reward and world-state SHA-256 digests plus final metrics. Trace copies and digest work are excluded from the timing rows. Optional same-chunk repeats test baseline reproducibility; on a diagnostic seed, the helper compares values as traces stream and records the first divergent step, every field's exact equality, and numeric mismatch count/maximum absolute difference. State detail includes RNG bytes and nonfloat fields such as alive flags and counters. A hash mismatch remains a failure; no numeric tolerance is applied. The JSON separates `timed_episodes`, cross-chunk `parity_episodes`, and `same_chunk_repeats`. The helper changes `sensor_chunk` only on an in-memory copy of the config and does not rewrite the run settings.

The CPU paired episode fixture uses 17 maps × 2 worms so chunks 4/8/16 exercise 5/3/2 raycast blocks. The long-body CPU fixture uses 17 × 8 for the same block-count distinction; the CUDA fixture uses 16 × 8 for 4/2/1 blocks. Focused checks cover three chunks, same-chunk repetitions, first-field numeric diagnostics, long bodies and deaths; the CUDA fixture remains skipped unless explicitly enabled. A CPU smoke run with the current champion (generation 3, genome 837), seed 71, and three policy steps matched per-step observation/action/reward/state digests and final metrics across chunks 4 and 16. It is a parity check, not throughput evidence.

```powershell
.\.venv\Scripts\python.exe -m slitherai.benchmark_observation --paired-policy --cuda-long-fixture --run runs/20260924-163112-852097 --seconds 15 --seeds 938271 1038282 --chunks 4 8 16 --diagnostic-seeds 1038282 --repeat-chunks 4 16 --device cuda
```


The CUDA fixture uses 16 maps with 8 worms, 96 body points, long bodies, mixed dead/alive worms, two clustered and two spread layouts repeated four times each. It checks bitwise observation parity and reports active points, alive/dead counts, per-observer capsule-candidate count distribution, time and peak allocation. The fixture has a CPU version in the focused tests; the CUDA test is opt-in through `SLITHERAI_RUN_CUDA_FIXTURE=1`.

## G5 CUDA result

Raw result: [chunk-policy-g5-2026-09-24T17-21-26-667Z.json](../../runs/20260924-163112-852097/analysis/chunk-policy-g5-2026-09-24T17-21-26-667Z.json). The run used champion generation 4/genome 1065, 64 maps, 15 simulated seconds, and common seeds 938271 and 1038282. Untraced episode wall-time medians were 9,249 ms for chunk 4 and 4,760 ms for chunk 16 (1.94× faster for chunk 16). The trace pass was separate from timing: actions and final metrics matched on both seeds, but state and reward digests differed on both. Observations matched on seed 938271 and differed on seed 1038282. This fails the exact per-step parity gate.

On the long-body fixture, observations stayed bitwise equal with 95 active points, 108 live/20 dead worms, and candidate counts from 54 to 603 across 70 distinct values. Chunk 4 took 14.71 ms and peaked at 108.20 MiB; chunk 16 took 23.69 ms and peaked at 359.74 MiB. This dense fixture is slower and uses more memory at chunk 16. Keep the active training setting unchanged pending diagnosis of the per-step mismatch and a new measured result. The pause window lasted 67.3 seconds; afterward the API reported `phase=training`, generation 5, `pause=false`, `stop=false`, `arena=0`.

## G6 diagnostic follow-up

Raw result: [chunk-policy-g6-diagnostic-20260924-193824-046157.json](../../runs/20260924-163112-852097/analysis/chunk-policy-g6-diagnostic-20260924-193824-046157.json). It used the saved champion at generation 5/genome 1065, PyTorch 2.8.0+cu129, seed 938271, the same 64-map/16-worm/15-second policy episode, and chunks 4/8/16. All cross-chunk traces matched bitwise for all 150 steps across observations, actions, states, rewards and final metrics. Repeated chunk-4 and chunk-16 episodes also matched their own first traces bitwise. This does not resolve the earlier G5 mismatch: it was not reproduced for this champion/seed, and the G5 capture did not save the first differing field.

This single-sample policy timing was 9,327.6 / 6,027.3 / 4,843.1 ms for chunks 4/8/16 (chunk 4 over 16: 1.93×). On the dense fixture, chunks 4/8/16 took 12.66 / 15.79 / 22.50 ms and peaked at 108.20 / 198.76 / 360.13 MiB; all observations were exact. The fixture had 108 live/20 dead worms, 95 active body points, and 54–603 capsule candidates. Together these measurements show the preferred tile depends on density; they do not establish a training-throughput or learning gain.

The pause window took 67.1 seconds. The only CUDA trainer was PID 20976 under its existing wrapper PID 14408; neither process was stopped or restarted. Pause was requested during generation 6, batch 1/4 (512/1280 episodes). The `finally` restoration verified `pause=false`, `stop=false`, `arena=0`; after three seconds the API reported training, generation 6, batch 2/4 (576/1280 episodes). No trainer or saved configuration setting was changed.

The timing report is an execution comparison, not a fitness baseline: it reports per-episode wall time, per-map score summaries and whether the two simulator variants stayed bitwise aligned. `sensor_chunk` currently participates in the saved `SimConfig` equality check during resume ([train.py:224-225](../../slitherai/train.py#L224)). Any production runtime override should record the effective chunk alongside the original saved setting and checkpoint, and should not alter `settings.json` in place. The offline helper already keeps its override separate; training, server and config code remain unchanged.

## Hot path and follow-up

`active_body_points()` reduces a CUDA tensor and converts the result to a Python integer ([sim.py:99-102](../../slitherai/sim.py#L99)). It is called from `_collisions()` once in each of the three physics substeps ([sim.py:117-120](../../slitherai/sim.py#L117)) and once again in `observe()` ([sim.py:232-235](../../slitherai/sim.py#L232)). Sensing also transfers `relevant.sum(-1).max()` to the host through `.item()` to size `topk` ([sim.py:245-253](../../slitherai/sim.py#L245)). These data-dependent extents likely impose about five host/device synchronization points per policy tick on CUDA. They are another exactness-preserving optimization target, but need their own benchmark because avoiding the transfers may process more masked capsules.

The first controlled diagnosis did not reproduce the G5 mismatch on seed 938271, so do not attribute its cause yet. In `WorldBatch._eat()`, `gain` is accumulated with floating-point `scatter_add_` ([sim.py:152](../../slitherai/sim.py#L152)); multiple foods may map to one worm, making this a plausible order-sensitive point. Collision kill counts use another `scatter_add_` ([sim.py:132](../../slitherai/sim.py#L132)); its addends are 0/1 counts. Observation food sectors use `scatter_reduce_(reduce='amax')` ([sim.py:280](../../slitherai/sim.py#L280)). The installed version is PyTorch 2.8.0+cu129. Its [deterministic algorithm list](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html) says CUDA `scatter_add_` can switch to a deterministic implementation under `torch.use_deterministic_algorithms(True)`; the [reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html) warn that determinism may reduce speed and the flag alone does not ensure full application reproducibility. The same docs require `CUBLAS_WORKSPACE_CONFIG` for deterministic CUDA `mm`/`mv`/`bmm`; the batched NEAT evaluator uses dense `torch.bmm` ([network.py:90](../../slitherai/network.py#L90)). PyTorch 2.8's deterministic operation list does not list CUDA `scatter_reduce_(reduce='amax')` among its deterministic alternatives, while the [scatter-reduce API](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.scatter_reduce_.html) warns that the operation may be nondeterministic on CUDA. These are candidate sources from documentation and code, not proof of the observed G5 discrepancy. If the mismatch recurs, the next controlled check should run in a separate process with determinism and its documented environment set before CUDA initialization, record any unsupported-op exception, and measure the speed impact; do not enable it in the live trainer.

### Bounded adaptive batching proposal

Keep the active and default chunk at 4. For a later opt-in test only, choose among 4/8/16 as an observation tiling parameter from a deterministic cost estimate using the existing `max_candidates`, ray count, worm count and observer count. The ray intermediate is approximately proportional to `sensor_chunk × worms × rays × max_candidates`; the existing candidate maximum is already computed in `observe()`. Calibrate a conservative scratch-memory cap, then choose the fastest measured tile among those under the cap for low-, medium- and high-candidate strata. Current evidence suggests smaller tiles for dense scenes and larger tiles for sparse scenes, but it does not yet determine safe thresholds.

Any prototype must process the same observers/capsules and leave both raw NEAT outputs unchanged; it may only choose the observation batch size. Require exact per-step observations, actions, rewards and simulator state on identical seeds, including chunk-boundary cases and long-body/death/crowding fixtures. Measure synchronized observation, full tick, full episode time and peak allocated bytes on representative density strata. A provisional adoption gate is bitwise equality on every fixture, no memory-cap violation, at least 10% paired median full-episode speedup overall, and no more than 5% median slowdown in any tested density stratum. Keep effective tile policy and thresholds in run provenance; do not change checkpoint compatibility or overwrite saved settings silently.

## G8 isolated adaptive-tile benchmark

Raw result: [adaptive-dense-budget-sweep.json](../../runs/20260924-163112-852097/analysis/adaptive-dense-budget-sweep.json), SHA-256 `FEAFA99D4F834FDC9DD88B5D4102347F6CC8B0C22B3BE2D1BA68D2A85A6E63CF`. The benchmark ran from a detached checkout at commit `439d04b7e06bae01fcc7122dc718bae5499dba97` (`C:/Docker/SlitherAI-perfbench-439d04b`) with the approved Python executable at `C:/Docker/SlitherAI/.venv/Scripts/python.exe`. Imports of `slitherai`, `sim`, `config`, `schema`, and `benchmark_observation` resolved to that detached checkout. Its focused CPU suite passed 8 tests; the CUDA-only test was skipped before the timed run.

The GPU run used PyTorch 2.8.0+cu129 on an RTX 4060 Laptop GPU and the original run's saved generation 7 champion, genome 1065 (`slither-neat-530-v1`). The original `settings.json` remained at `sensor_chunk=4`. Training was paused through port 8765 from `pause=false, stop=false, arena=0`; the pause was acknowledged during generation 8, game 3/5, batch 3/4. The benchmark took 54.7 seconds including pause acknowledgement and restoration. `finally` restored `pause=false`; `stop=false` and `arena=0` were preserved. The API returned to active training at generation 8, game 3/5, batch 3/4; a follow-up read showed batch 4/4. The trainer wrapper/child PIDs 20948/8028 remained in place. No training process was stopped or restarted and no settings were edited.

The paired policy run used 64 maps × 16 worms, seed 938271, and 10 simulated seconds (100 steps). All observations, actions, state and reward digests plus summary metrics matched exactly for every adaptive budget. Candidate counts ranged from 20 to 109 (p50 75, p90 101). Budget 1 fell back to chunk 4 on all 100 calls and exceeded its requested work budget on all calls, as the documented smallest-tile fallback requires. Budgets 4M, 7M and 14M all selected chunk 16 for all 100 calls; the maximum estimated work was 2,427,648 elements, so this sparse policy episode never exercised chunk 8.

The single untraced full-episode sample took 6,220.4 ms at fixed chunk 4. Adaptive timings were 6,030.6 ms for budget 1, 3,165.7 ms for 4M, 3,149.7 ms for 7M and 3,157.1 ms for 14M. These one-seed, one-sample timings are a smoke measurement and do not establish a stable throughput gain.

The dense fixture used 16 maps × 8 worms, with the same seeded initial state for all fixed and adaptive conditions. It had 95 active body points, 108 live and 20 dead worms, and 54–603 candidates per observer (median 210, 70 distinct values). Each condition received one warmup and five synchronized observation repeats in alternating order. The parity/profile pass ran outside timing. Every row matched the same reference observation bit-for-bit, and each fixture's initial tensors matched exactly.

| Condition | Selected chunk | Median observe time | Peak allocated | Exact parity |
|---|---:|---:|---:|---|
| Fixed chunk 4 | 4 | 15.23 ms | 108.20 MiB | yes |
| Fixed chunk 8 | 8 | 14.59 ms | 198.76 MiB | yes |
| Fixed chunk 16 | 16 | 19.01 ms | 360.13 MiB | yes |
| Adaptive budget 1 | 4 (fallback) | 14.63 ms | 108.20 MiB | yes |
| Adaptive budget 4M | 8 | 14.71 ms | 198.76 MiB | yes |
| Adaptive budget 7M | 16 | 20.21 ms | 360.13 MiB | yes |
| Adaptive budget 14M | 16 | 18.54 ms | 360.13 MiB | yes |

On this dense fixture, budget 4M selected chunk 8 at 3,357,504 estimated elements; 7M selected chunk 16 at 6,715,008. Chunk 16 took 6.3% longer than fixed chunk 16 at budget 7M and used 3.3× the peak allocation of chunk 4. Five repeats still leave timing noise, but this rejects a blanket chunk 16 default and confirms that these thresholds reach different tiles on dense versus sparse workloads. The adaptive policy preserved exact observations in both workloads. No checkpoint, trainer config, server code or active settings changed.

Next confirmation, not run here: after generation 10 validation, compare fixed chunk 4 with budget 4M on the same saved champion for 64 maps × 16 worms × 90 simulated seconds, using seed 1038282, separate untraced timings and exact traces. Reserve a dedicated pause window capped at 240 seconds and enforce a 210-second benchmark subprocess timeout. Keep this as benchmark-only runtime context with recorded threshold provenance; any trainer integration must coordinate with the sensor-mode work and preserve the legacy chunk 4 default.

## Input and action contract

The schema is consistent: 87 rays × 6 channels + 8 global values = 530 inputs, with two outputs ordered as boost probability then absolute direction in turns ([schema.py:5-11](../../slitherai/schema.py#L5)). The network returns the first two output nodes ([network.py:87-92](../../slitherai/network.py#L87)); `step()` thresholds boost at 0.5 and wraps the direction with modulo before taking the shortest angular delta ([sim.py:165-173](../../slitherai/sim.py#L165)). Thus direction values 0 and 1 denote the same heading; there is no 530/2 shape or wraparound execution mismatch.

A one-seed CPU probe of 16 initial genomes on 16 initial observations produced output values from 0.092 to 0.903, with no outputs below 0.01 or above 0.99. That small sample gives no evidence of blanket initial sigmoid saturation; it is not a population-wide saturation analysis.
