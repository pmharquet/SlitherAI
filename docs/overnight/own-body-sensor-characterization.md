# Own-body sensor characterization

## Scope and evidence

This is a CPU-only, synthetic comparison of the extension sensor and the training simulator. It executes the real `compute()` in `extension/lidar-main.js` and `WorldBatch.observe()` in `slitherai/sim.py` on the same head position, heading, sampled centerline points, and radius. It does not use the missing gameplay JSONL export, alter sensor behavior, or make a claim about how often either geometry occurs in real games.

The shared fixture uses a head at `(0, 0)`, heading east, body sample spacing `9`, and a simulator body radius of `15.154` (mass `35`). The extension's `sc` is set so its estimated radius is the same `15.154`, which isolates own-body geometry semantics from radius calibration. The inspected ray is the rear shoulder at `-166°`; both sensors use the same 87-ray angle layout and `1 / (1 + distance / 100)` proximity transform.

## Code locations

| Implementation | Location | Own-body rule |
| --- | --- | --- |
| Browser extension | [lidar-main.js](../../extension/lidar-main.js#L330) | Builds capsules from `own.pts`; it uses radius `2 × ownRadius + 3` (line 332), and excludes vertices according to Euclidean head distance `< 2 × ownRadius` (lines 335–346). |
| Training simulator | [sim.py](../../slitherai/sim.py#L334) | Own capsules use the physical model radius (line 338); it includes a segment when its sample index times body spacing is at least `3 × radius` (lines 340–341). The criterion is distance along the stored body, not spatial distance to the head. |

Both use rounded capsule ray intersections after candidate filtering. The extension's `own.pts` do not include an explicit head capsule; the simulator forms segments from `body[k]` to `body[k+1]`, with `body[0]` at the head in this fixture.

## Measured synthetic cases

| Shared centerline | Extension self-body return | Simulator self-body input | Difference |
| --- | ---: | ---: | --- |
| Straight tail, points every 9 units | Distance `2.781`, proximity `0.973` | Distance-equivalent `44.715`, proximity `0.691` | Extension first admits the center at 36 units (outside its `2r = 30.309` head cutoff) and inflates it to `2r + 3 = 33.309`. Simulator first admits segment index 6, beginning at 54 units because `6 × 9 ≥ 3r = 45.463`, and uses radius `r = 15.154`. |
| Near-head coil: 27 units straight, then a radius-2 loop around `(-25, 0)` | No return | Distance-equivalent `8.671`, proximity `0.920` | Every coil point is within 27 units of the head, so the extension drops it under its spatial cutoff. The simulator retains late-arc segments because their along-body index exceeds `3r`, despite their spatial proximity to the head. |

Distance-equivalent for the simulator is reconstructed from its observed proximity using `100 × (1/value − 1)`. The path in the coil case is deliberately tight and is a semantic stress test, not evidence that this exact shape is common or physically typical. The test asserts the straight case against analytic ray-circle endpoint distances and checks that the coil produces an extension omission while the simulator still senses the late arc.

Reproduce with:

```powershell
python -m pytest -q tests/test_own_body_sensor_fidelity.py
```

The test bridge at [own_body_sensor_probe.cjs](../../tests/own_body_sensor_probe.cjs) invokes the real JavaScript extension. The Python test builds the same sampled points into a two-worm CPU `WorldBatch`; the second worm and food are outside the selected geometry.

## Intent and compatibility

The radius and cutoff behavior is present in the initial public-release commit (`3094b71`). The repository has no earlier implementation history from which to infer a previous deliberate change. The code comments explain the estimated extension size radius and the general lidar validation plan, but neither the comments nor [lidar.md](../lidar.md) document why own-body sensing uses an inflated `2r+3` radius and a spatial `2r` exclusion while training uses physical radius and an along-body `3r` exclusion. Intent therefore remains unknown; the visible code is not evidence that this difference was a chosen training-transfer contract.

## Candidate versioned correction

For a future transfer-calibrated contract, define a new sensor/schema version that keeps legacy recordings and checkpoints identifiable, estimates a single own-body sensing radius, and applies the simulator's `3r` along-body exclusion to the extension's ordered centerline. Keep the current implementation labeled as the legacy version. Before enabling the new version, verify the proximal-to-distal order and point spacing from an actual export, calibrate the extension size estimate against simulator radius, and rerun the straight/curved synthetic parity tests plus multi-zoom real-game checks.

The input tensor would keep the same number of values, so old genomes could still load while silently interpreting a changed fourth channel. Bump the observation/schema identity and record it in datasets, holdout outputs, and checkpoints; do not mix legacy and new sensor versions in training, candidate scoring, or final holdout comparison. No versioned correction is deployed here.
