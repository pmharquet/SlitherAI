# Performance and observation review

## Finding

The strongest measured exact-preserving tuning candidate is `sensor_chunk=16` for a new run. In `WorldBatch.observe()`, each raycast block covers `sensor_chunk * worms` observers ([sim.py:257](../../slitherai/sim.py#L257)); raising the chunk from 4 to 16 cuts that loop from 16 blocks to 4 for the default 64 maps and 16 worms. The sensor math and the 530 output values stayed bitwise equal in the bounded CUDA sweep.

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

## Hot path and follow-up

`active_body_points()` reduces a CUDA tensor and converts the result to a Python integer ([sim.py:99-102](../../slitherai/sim.py#L99)). It is called from `_collisions()` once in each of the three physics substeps ([sim.py:117-120](../../slitherai/sim.py#L117)) and once again in `observe()` ([sim.py:232-235](../../slitherai/sim.py#L232)). Sensing also transfers `relevant.sum(-1).max()` to the host through `.item()` to size `topk` ([sim.py:245-253](../../slitherai/sim.py#L245)). These data-dependent extents likely impose about five host/device synchronization points per policy tick on CUDA. They are another exactness-preserving optimization target, but need their own benchmark because avoiding the transfers may process more masked capsules.

The next useful measurement is a seeded representative episode with grown bodies and deaths, comparing chunks 4, 8 and 16 on observation time, full tick time, peak memory and per-episode throughput. Use the current checkpoint after the generation 5 validation boundary; avoid changing the active run's configuration. `sensor_chunk` is stored in `settings.json` and resume checks compare the saved config ([train.py:224-225](../../slitherai/train.py#L224)), so adopting a different value on a resumed run needs explicit config/provenance handling. The short benchmark justifies that follow-up, not changing training settings mid-run.

## Input and action contract

The schema is consistent: 87 rays × 6 channels + 8 global values = 530 inputs, with two outputs ordered as boost probability then absolute direction in turns ([schema.py:5-11](../../slitherai/schema.py#L5)). The network returns the first two output nodes ([network.py:87-92](../../slitherai/network.py#L87)); `step()` thresholds boost at 0.5 and wraps the direction with modulo before taking the shortest angular delta ([sim.py:165-173](../../slitherai/sim.py#L165)). Thus direction values 0 and 1 denote the same heading; there is no 530/2 shape or wraparound execution mismatch.

A one-seed CPU probe of 16 initial genomes on 16 initial observations produced output values from 0.092 to 0.903, with no outputs below 0.01 or above 0.99. That small sample gives no evidence of blanket initial sigmoid saturation; it is not a population-wide saturation analysis.
