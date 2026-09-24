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

## Implemented simulator mode: `export-v1`

The simulator now has an explicit `sensor_version`: `legacy-v1` preserves the historical behavior and remains the default; `export-v1` changes only the own-body ray channel. For a shared ordered point list and base radius `r`, the new mode follows the extension's existing construction:

1. Use the extension's current effective radius rule `clamp(14.5 × sc, 6, 80)` as the source reference; in a simulator fixture, supply the same `r` explicitly or use the simulator radius only when it has been calibrated to that reference.
2. Retain vertices whose Euclidean distance from the head is at least `2r`.
3. Connect adjacent retained vertices with a capsule when their separation is below `600`, with capsule radius `2r + 3`.
4. Add a circle of radius `2r + 3` for each retained vertex that is not connected to an accepted capsule.
5. For each ray, take the nearest in-range intersection and apply the existing `1/(1+d/100)` transform.

The simulator preserves stored adjacency and uses float32, so parity with the JavaScript double arithmetic is numeric rather than bit-for-bit. CPU tests execute the actual extension `compute()` and compare all 87 own-body ray values for shared straight and curved centerlines, plus isolated vertices and gaps over 600 units; the asserted absolute tolerance is `3e-5`. The actual game-to-simulator mapping remains unverified. Extension `own.pts` are filtered for invalid/`dying` entries, use fractional `fx`/`fy` coordinates when present, and can have irregular gaps; the simulator stores a fixed-spacing resampled body. Most importantly, source `r` is estimated from `sc`, while simulation `r` comes from mass. The shared tests intentionally set these values equal and cannot establish that they match in a live game. Viewport half-extents also need separate calibration.

This route preserves the extension's present behavior and avoids tuning it to favor the simulator. `export-v1` changes only the simulator's `self_body` channel under an opt-in mode and keeps all **530 inputs** and NEAT input keys. It is still a real observation-distribution change: the larger radius and different geometric filter alter ray values, and the extra disconnected-vertex circles may change sensor candidate counts. Re-benchmark representative grown bodies and repeat CPU/CUDA parity and fixed validation before training in that mode. This implementation task did not activate `export-v1` in any training run.

## Configuration, schema, resume, and holdout implications

| Surface | Required design if `export-v1` is approved |
| --- | --- |
| Simulation config | Implemented in [config.py](../../slitherai/config.py): the enum is validated, saved in `settings.json`, and `SimConfig.from_dict()` maps a missing old field to `legacy-v1`. Resume compares normalized settings. |
| New run and API | Implemented in [train.py](../../slitherai/train.py) and [server.py](../../slitherai/server.py): CLI and start options allow deliberate selection on a fresh run; resume derives the mode from saved settings. No running legacy session was changed. |
| Input schema | Implemented in [schema.py](../../slitherai/schema.py), [network.py](../../slitherai/network.py), and [dataset.py](../../slitherai/dataset.py). The 530 inputs and NEAT keys are preserved, while schema identity/metadata record own-body semantics. |
| Resume and history | Resume rejects mode changes. A present unreadable or unknown `schema.json` now fails closed before metadata is rewritten. Only a historical run with no schema file and no `sensor_version` in saved config migrates as legacy; explicit-version settings require a recognized schema. |
| Holdout | Implemented in [evaluate_holdout.py](../../slitherai/evaluate_holdout.py): checkpoint training mode is recorded separately from evaluation mode. Cross-mode rescoring requires a complete explicit comparison config and freshly scores every policy; cached scores only select a champion. |

The holdout requires a complete explicit comparison config to cross sensor modes; `sensor_version` participates in its config and schema identity. The existing `common-reference-v2` protocol remains the anchor/opponent aggregation definition, with sensor semantics recorded separately.

The mode preserves the observed extension algorithm as an opt-in training target without changing future recorded inputs. The principal limits remain radius/body-point/view calibration and added sensor-work cost. Existing runs, checkpoints, and historical validation stay legacy. Before any experiment is run with `export-v1`, review the code and then compare it against legacy on common validation and an independent holdout; the synthetic parity test alone does not establish transfer benefit.

## Future warm-start boundary (not implemented)

If a later reviewed experiment wants to reuse a legacy population, it should create a **new** run configured as `export-v1` and import genotype genes from an explicitly named trusted checkpoint. It must not call strict `--resume`: resume continues one objective and remains mode-strict. The new run should record the source checkpoint path/hash and generation as initialization provenance, while retaining its own export schema, settings, protocol, and fresh run identity. The old run's files and settings stay untouched.

The import should carry genome structure/weights (and only the NEAT bookkeeping required to continue those genes), then clear every cached fitness/anchor fitness, assign a fresh generation zero, and rebuild species/stagnation state under the new run. Do not copy source history, episode scores, fixed validation, baselines, champion rankings, or species fitness/stagnation counters: they were measured under legacy sensing and cannot serve as export-v1 evidence. Establish a fresh generation-zero evaluation and fresh validation/baselines before comparing the warm-start against a newly initialized export-v1 control. Keep an explicit source-mode field in provenance. A future `--init-from` command needs its own compatibility checks and tests for topology, innovation bookkeeping, reset fitness, and no writes into the source run; this task adds no transfer/import path.
