# Source-game fidelity audit

Date: 2026-09-24. This audit reads the extension and simulator source and the active run settings. It does not change simulation or training code.

## Export availability

The requested file `C:\Users\88mat\Downloads\slitherai-2026-09-23T15-09-52-803Z-b2039f8f.jsonl` is absent on this workspace host. A bounded filename search under the repository, Downloads, Documents, and Desktop also found no matching September 23 export, including the older 14:37 name. I therefore did not calculate source-side sample statistics or present estimates as new measurements.

The earlier [training notes](../training.md) mention an export-derived arena-radius estimate near 31,899 units. Its input export is unavailable here, so this report treats that number as historical and unverified. The reproducible analyzer in [source_fidelity.py](../../slitherai/source_fidelity.py) accepts the missing export and emits aggregate JSON and Markdown when it is accessible.

```powershell
.venv\Scripts\python.exe -m slitherai.source_fidelity `
  C:\Users\88mat\Downloads\slitherai-2026-09-23T15-09-52-803Z-b2039f8f.jsonl `
  --settings runs\20260924-163112-852097\settings.json `
  --out runs\20260924-163112-852097\analysis\source-fidelity
```

## Active simulator reference

The current run settings are from `runs/20260924-163112-852097/settings.json`: 64 maps, 16 worms, 1,024 ambient-food slots with 8 moving prey, arena radius 2,400 ±20%, `dt=0.1`, view half-extents 600×500, base/boost speed 115.8/210, `raw_speed_scale=20`, turn rate 2.8 rad/s, initial mass 35, and boost cost 3 mass/s.

At the nominal initial mass, the simulator radius formula gives `14.5×(1+35/300)^0.4 = 15.154` units. The configured initial mass range yields radii 15.027–15.280. At nominal radius, the size speed factor is 1.0045, giving 116.32 base and 210.95 boost units/s. The speed ratio is 1.813. The simulator's initial turn cap is 2.739 rad/s, or 156.93°/s; it falls as simulated radius grows. Boost costs 0.3 mass per 0.1-second action, or 3 mass/s while continuously active.

The initial nominal view spans about 1,226.8×1,022.3 world units after the radius zoom factor. The local food density before respawns, clustering, deaths, and boost deposits is 56.59 items per million arena units² at radius 2,400. The configured radius variation gives a uniform-density reference of 39.30–88.42 items per million units² across the radius extremes. The simulation begins with 15 fixed heuristic opponents per focal worm.

If the historical 31,899-unit estimate were confirmed, it would be 13.29 times the local nominal arena radius (11.08–16.61 times the local radius extremes). That scale difference could affect encounter frequency, but arena size alone cannot prescribe food density, player count, or camera settings.

## Sensor and action contract

| Component | Extension export | Simulator/importer | Fidelity assessment |
|---|---|---|---|
| Ray layout | 87 head-relative rays from −166° to +166°, nonuniform spacing, no direct rear ray | Same ordered 87 angles | Exact angle/count contract; the analyzer checks recorded rays against it. |
| Channels | Visible range, enemy head, enemy body, own body, border, food/prey | Six channels in the same order; prey shares food | Same shape and broad meanings. The geometry feeding several channels differs below. |
| Range/proximity scaling | Per-ray camera cutoff; proximity `1/(1+d/100)` | Range `r/(r+1000)`; proximity `1/(1+d/100)` with absent hits set to zero | Encoding formulas match. A browser's camera rectangle does not necessarily match the simulator's virtual view. |
| Food signal | `min(size/20,1)×proximity`; client food `sz`, estimated ray entry | Same feature form; size and entry derived from simulated food mass/size | Formula matches, object-size meaning and food population do not. |
| Enemy head/body margin | Estimated own and enemy radii plus 3 units | Simulated own and enemy radii plus 3 units | Same margin; radius sources are not calibrated to the server collision radius. |
| Own-body signal | Self capsule radius `2×estimated_radius+3`; skips geometry within about `2×radius` of the head | Uses own radius; ignores body capsules before the `3×radius` along-body cutoff | **Real observation mismatch:** self-body proximity, thickness, and near-head cutoff differ. It changes an input channel used by the policy. |
| Border signal | Ray exit from estimated circle `0.98×client grd − estimated own radius − 3` | Ray exit from configured arena radius − simulated radius − 3, censored by local view | Algebra and 3-unit inset are similar. Boundary estimate, center convention, and world scale remain unvalidated. |
| Direction | Records cursor steering angle relative to current heading; importer adds heading and maps the absolute target to turns `[0,1)` | Network emits absolute direction turns; simulation pursues that heading | Coordinate convention agrees: east 0, positive y/down 0.25, west 0.5, north 0.75. The browser label is a sampled pointer target, not a replay of held game inputs. |
| Boost | Records whether mouse button or Space is held at sample time | Activates at output ≥0.5 if mass exceeds the floor; spends mass and deposits it behind the worm | Input request shape matches. Export does not indicate that boost actually engaged or how much mass it consumed. |

The extension's sample interval is adaptive: the page schedules another read after `max(100ms, min(750ms, 8×compute_ms))`. Thus observations and action labels are asynchronous snapshots, not a fixed 10 Hz action trajectory. The analyzer uses browser `performance.now()` deltas where available, reports gaps, and keeps movement intervals no longer than 0.3 s for its speed-scale estimate.

## What the export can measure

The script reports distributions and sample counts, without retaining individual rows:

- **Speed scale:** short head-position displacement divided by elapsed seconds and previous `speedRaw`, grouped by stable sampled boost request. It filters for ≤0.3 s intervals, ≤10% `speedRaw` change, and ≤15° heading change. Curved paths, collisions, interpolation, server lag, or a mid-interval control change can still bias the ratio. It is an empirical scale check against the simulator's 20 world units per raw unit, not a verified movement law.
- **Turning:** wrapped heading change divided by elapsed time, plus a subset whose previous `wantedHeading` error is at least 30°. These are human-chosen turn rates. Even large requested errors do not prove the user held a constant target, so they cannot identify maximum turn rate.
- **Size:** distributions of `bodyRadiusEstimate`, `scaleRaw`, and segment count, plus a check that the estimate follows the extension's clamped `14.5×scaleRaw` rule. The estimate is not a server-measured collision radius. Public score, segment count, and `sc` do not identify simulator mass or its growth curve.
- **Boost:** sampled held-input share and `speedRaw`/movement-speed distributions by requested input. There is no mass, boost-active flag, or mass-spend counter; boost cost is not identifiable from these fields.
- **View and density:** world-space camera width/height from the recorded canvas rectangle and zoom; visible food/prey per viewport area; and available server-player counts. These are viewport-clipped local counts, affected by food clusters, camera motion, browser dimensions, and other players. They do not identify global source-game density.
- **Contract check:** ray count and angle mismatches, reconstructed 530-input validity, nonzero channel fractions, range/censor summaries, action-target alignment against `wantedHeading`, and decoded return-kind counts.

## Readout and next calibration priorities

The strongest confirmed transfer issue from code is the **own-body channel geometry mismatch**. Preserve it as an explicit test/calibration target before interpreting behaviors around self-coils. The view channel also needs actual export statistics because the source uses the real camera rectangle and local training uses fixed half-extents scaled by simulated radius.

Use timestamped displacement to check `raw_speed_scale=20` only after the export is restored. Keep source turn observations separate from maximum-turn physics. Do not tune boost cost from button state or infer mass from public score. If the historical arena estimate is reconfirmed, treat arena scale and encounter/density setup as separate calibration questions rather than scaling all settings by the same factor.

Reproduce the numeric analysis with the command in the analyzer documentation / CLI help, supplying the actual export path and active `settings.json`. The analyzer is read-only with respect to the export and training run; it writes only the two files named by `--out`. Synthetic CPU tests cover timing, sensor shape, view, speed-scale math, and partial/corrupt exports. No real-user export or GPU was used for this report.
