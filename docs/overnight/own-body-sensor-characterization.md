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

## Simulator-side alternative: `export-v1`

The more directly evidence-based experiment is to leave the observed extension contract alone and add a simulator mode that reproduces its existing own-body construction. Name the current simulator behavior `legacy-v1` and keep it as the default; make `export-v1` explicit for a new run. On a shared ordered vertex list and a shared base radius `r`, that mode can reproduce the extension's own-body shapes:

1. Use the extension's current effective radius rule `clamp(14.5 × sc, 6, 80)` as the source reference; in a simulator fixture, supply the same `r` explicitly or use the simulator radius only when it has been calibrated to that reference.
2. Retain vertices whose Euclidean distance from the head is at least `2r`.
3. Connect adjacent retained vertices with a capsule when their separation is below `600`, with capsule radius `2r + 3`.
4. Add a circle of radius `2r + 3` for each retained vertex that is not connected to an accepted capsule.
5. For each ray, take the nearest in-range intersection and apply the existing `1/(1+d/100)` transform.

The extension rule is deterministic and can be reproduced for any supplied ordered point list without assuming whether the source lists head-near points first or last: preserve the supplied adjacency and use the same numeric inputs. It is not bit-for-bit arithmetic parity: the browser uses JavaScript doubles and the simulator uses Torch float32. The actual game-to-simulator mapping remains unverified. Extension `own.pts` are filtered for invalid/`dying` entries, use fractional `fx`/`fy` coordinates when present, and can have irregular gaps; the simulator stores a fixed-spacing resampled body. Most importantly, source `r` is estimated from `sc`, while simulation `r` comes from mass. The shared test intentionally sets these values equal and cannot establish that they match in a live game. Viewport half-extents also need separate calibration.

This route preserves the extension's present behavior and avoids tuning it to favor the simulator. It changes only the simulator's `self_body` channel under an opt-in mode and keeps all **530 inputs** and NEAT input keys. It is still a real observation-distribution change: the larger radius and different geometric filter alter ray values, and the extra disconnected-vertex circles may change sensor candidate counts. Re-benchmark representative grown bodies and repeat CPU/CUDA parity and fixed validation before training in that mode. No simulator code or mode has been added in this task.

## Configuration, schema, resume, and holdout implications

| Surface | Required design if `export-v1` is approved |
| --- | --- |
| Simulation config | Add an explicit `sensor_version` with `legacy-v1` and `export-v1`; validate the enum and save it in `settings.json`. `SimConfig.from_dict()` must map a missing field in old settings to `legacy-v1`. Normalize old config dictionaries before strict resume equality checks, since `Trainer.train()` currently compares saved config JSON directly to `dataclasses.asdict(config)` ([train.py](../../slitherai/train.py#L237)). |
| New run and API | Expose a deliberate selection for a fresh run. Resume must restore the saved mode rather than use a new-run default. The current resume API reconstructs CLI arguments from saved fields ([server.py](../../slitherai/server.py#L87)); it will need to preserve `sensor_version`. The current run `20260924-163112-852097` has no such field and schema `slither-neat-530-v1`, so treat it as `legacy-v1` and never switch it in place. |
| Input schema | Keep the ray count, six channels, globals, outputs, and NEAT topology at 530 inputs. Give the two channel semantics distinct schema identities/contracts while retaining the current identity as legacy; a dimension-only check cannot catch a changed fourth channel. Store the mode in `schema.json`, champion payloads, network/preview metadata, and converted-dataset reports. `network.load_config()` can keep its 530-key shape check. |
| Resume and history | A mode change must reject continuation in either direction. Anchor selection fitness, fixed validation, and cached histories were produced under one observation mode; appending generations under another would mix objectives. An old settings file without the field resolves to legacy, preserving checkpoint behavior. |
| Holdout | Make schema validation recognize both saved source schemas. Record the checkpoint's training mode separately from the selected evaluation mode. A cross-mode evaluation is allowed only through an explicit complete comparison config and must freshly rescore every policy; historical scores may identify which genome to load but must not be compared as holdout results. Include `sensor_version` in config/schema hashes and holdout metadata. Current holdout schema checks accept only one global `VERSION`/`contract()` ([evaluate_holdout.py](../../slitherai/evaluate_holdout.py#L52)); its metadata also reports that fixed version ([lines 144–181]). |

The holdout's existing complete-config requirement will naturally require the new field after it is added. Keep `common-reference-v2` for the anchor/opponent aggregation only if it remains that narrow definition; the sensor mode must still be recorded and checked separately. Alternatively, include the mode in protocol metadata so resume and comparison gates fail closed.

The principal advantage over changing the extension is that the known extension algorithm becomes the training target, without changing future recorded inputs or inventing an extension behavior unsupported by the absent export. The principal limits are radius/body-point/view calibration and added sensor-work cost. The implementation should first preserve legacy outputs, then pass an all-ray extension-oracle parity test on shared points in `export-v1`; only after that should a new run compare legacy and export-v1 on common validation and an independent holdout. Existing runs, checkpoints, and historical validation stay legacy.
