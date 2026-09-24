# Offline frozen-policy comparison

`slitherai.evaluate_holdout` replays frozen NEAT policies with the production `evaluation.play_episode()` loop. It adds a fresh initial-generation champion, each requested candidate, and the built-in heuristic and circle policies. Every policy gets the same seed, saved simulation settings, independent maps, focal slots (`map_index * 7 % worms`), and heuristic opponents. No training state or settings are changed.

Historical episode `score` is used only to choose a generation's champion: the tool takes its argmax in `episodes/generation-NNNN.json`, then finds that genome in `checkpoint-N`. It does not use checkpoint fitness or cached validation fitness as an outcome. Every reported fitness, food gain, survival, age, kill, death, boost, turn, and reward measure comes from a new simulation. The initial policy is generation 0.

## Candidate selection

Use validation results to decide which candidate to compare before running the holdout. Freeze the candidate list before using the reserved final seed. A generation candidate is that generation's training champion; a payload candidate can be a saved validation champion such as `best-validation.pkl`.

Example with the active run and an already selected generation:

```powershell
.venv\Scripts\python.exe -m slitherai.evaluate_holdout `
  --run runs\20260924-163112-852097 `
  --candidate-generation 9 `
  --seed 741852963 --maps 64 --seconds 90 --device cuda
```

The `--seed`, `--maps`, and `--seconds` arguments are mandatory, and duration must align to the saved `dt`. For a saved model payload, use `--candidate-payload best_validation=RUN\best-validation.pkl`. Add either candidate option more than once to compare a short, preselected list. The tool always adds generation 0, heuristic, and circle. Pickle checkpoints and payloads must be trusted local run artifacts.

The seed `741852963` with 64 maps and 90 seconds is reserved for the final morning comparison. Do not use it for smoke tests or exploratory candidate selection. Run the final suite once after training and model selection are complete; do not tune candidates against its result.

## Config compatibility

By default, the tool uses the run's saved `SimConfig` and requires its saved protocol and reward version to match the active evaluator. It rejects runs whose observation contract differs. Candidate payloads inside the selected run inherit that run's saved simulation/reward context. A payload from outside the run requires `--comparison-config`.

An explicit comparison config lets all policies be replayed under one declared simulator and protocol, including when the source run has an older reward. It must be JSON with every `SimConfig` field and the exact active protocol object, for example:

```json
{
  "config": {
    "maps": 64,
    "worms": 16,
    "foods": 1024,
    "preys": 8,
    "body_points": 96,
    "arena_radius": 2400.0,
    "arena_variation": 0.2,
    "view_half_width": 600.0,
    "view_half_height": 500.0,
    "dt": 0.1,
    "substeps": 3,
    "base_speed": 115.8,
    "boost_speed": 210.0,
    "raw_speed_scale": 20.0,
    "turn_rate": 2.8,
    "initial_mass": 35.0,
    "min_mass": 10.0,
    "boost_cost": 3.0,
    "body_spacing": 9.0,
    "sensor_chunk": 4,
    "reward_version": "growth-v2"
  },
  "protocol": {
    "version": "common-reference-v2",
    "anchor_games": 4,
    "rotating_games": 1,
    "anchor_weight": 0.8,
    "aggregate": "half_mean_half_median",
    "opponents": "fixed_food_and_avoidance",
    "validation_maps": 32,
    "validation_seconds": 90
  }
}
```

The command's `--maps` value replaces `config.maps`; all other config values are fixed by the explicit file. An observation-schema mismatch is always rejected because the current network cannot interpret a different input contract. The recorded config, protocol, reward, and schema each have SHA-256 provenance fields.

## Results and limits

The tool writes a compact JSON file with per-map metrics and all paired model differences, plus a Markdown summary, under `RUN/analysis/holdout/` by default. Its paired standard errors and fixed-seed bootstrap intervals measure variation across this declared map suite. They are not evidence of performance across unseen seeds, opponent policies, or the source game. Seeded maps are the unit of resampling; the reported intervals do not account for model-selection uncertainty.

This is an offline evaluator and imports no external APIs. CPU is the default; `--device cuda` opts into the configured CUDA device. Tests use a tiny synthetic CPU run and verify checkpoint champion provenance, fresh rescoring, explicit old-reward handling, and deterministic paired uncertainty. They do not run the reserved seed suite.
