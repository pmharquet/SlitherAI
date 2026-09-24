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

The timing report is an execution comparison, not a fitness baseline: it reports per-episode wall time, per-map score summaries and whether the two simulator variants stayed bitwise aligned. `sensor_chunk` currently participates in the saved `SimConfig` equality check during resume ([train.py:224-225](../../slitherai/train.py#L224)). Any production runtime override should record the effective chunk alongside the original saved setting and checkpoint, and should not alter `settings.json` in place. The offline helper already keeps its override separate; training, server and config code remain unchanged.

## Hot path and follow-up

`active_body_points()` reduces a CUDA tensor and converts the result to a Python integer ([sim.py:99-102](../../slitherai/sim.py#L99)). It is called from `_collisions()` once in each of the three physics substeps ([sim.py:117-120](../../slitherai/sim.py#L117)) and once again in `observe()` ([sim.py:232-235](../../slitherai/sim.py#L232)). Sensing also transfers `relevant.sum(-1).max()` to the host through `.item()` to size `topk` ([sim.py:245-253](../../slitherai/sim.py#L245)). These data-dependent extents likely impose about five host/device synchronization points per policy tick on CUDA. They are another exactness-preserving optimization target, but need their own benchmark because avoiding the transfers may process more masked capsules.

Next, run the documented 4/8/16 sweep with same-chunk repeats for 4 and 16 on seed 1038282. The per-event numeric comparison will show whether chunk 4 and chunk 16 are each reproducible, and will locate their earliest cross-chunk state difference. The G5 raw JSON above remains the provenance record for the completed run; write this diagnostic result to a new timestamped JSON file rather than replacing it. If a repeated same-chunk trace differs, investigate the simulator's CUDA reduction/scatter behavior before interpreting cross-chunk deltas. If same-chunk traces are stable and only cross-chunk differs, inspect the first reported field and timestep. `sensor_chunk` is stored in `settings.json` and resume checks compare the saved config ([train.py:224-225](../../slitherai/train.py#L224)), so any eventual override needs explicit config/provenance handling. The offline helper keeps its chunk override separate and does not edit the active run settings.

## Input and action contract

The schema is consistent: 87 rays × 6 channels + 8 global values = 530 inputs, with two outputs ordered as boost probability then absolute direction in turns ([schema.py:5-11](../../slitherai/schema.py#L5)). The network returns the first two output nodes ([network.py:87-92](../../slitherai/network.py#L87)); `step()` thresholds boost at 0.5 and wraps the direction with modulo before taking the shortest angular delta ([sim.py:165-173](../../slitherai/sim.py#L165)). Thus direction values 0 and 1 denote the same heading; there is no 530/2 shape or wraparound execution mismatch.

A one-seed CPU probe of 16 initial genomes on 16 initial observations produced output values from 0.092 to 0.903, with no outputs below 0.01 or above 0.99. That small sample gives no evidence of blanket initial sigmoid saturation; it is not a population-wide saturation analysis.
