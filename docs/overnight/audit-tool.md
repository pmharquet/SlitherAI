# Overnight run audit

`slitherai.overnight_audit` produces a compact JSON report and a short Markdown report from a training run's JSON telemetry. It reads `history.jsonl`, `status.json`, settings, validation results, species summaries, and checkpoint filenames/stat metadata. It does not load checkpoint or genome pickle files, read the per-genome episode matrices, call external services, or change training configuration.

Run it from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m slitherai.overnight_audit `
  --run runs\20260924-163112-852097 `
  --deadline 2026-09-25T08:00:00+02:00
```

By default, output is written to `RUN/analysis/overnight/audit.json` and `audit.md`. Pass `--out PATH` to choose another report directory. The deadline must include a timezone. Omit it when there is no fixed wall-clock budget.

The report compares selection population means with fixed-anchor means, summarizes food gain, survival, deaths, boost use and turning, lists fixed-validation history with its reported standard error, and reports current species count, effective diversity, tiny species, births/removals, and compatibility threshold. The selection mean comes from each history row's `mean`; `evaluation.fitness` is retained separately as the raw episode outcome mean. Pace uses the median `seconds` field from up to five completed generations. Pauses during those evaluations are included in the recorded durations. The remaining-time estimate omits validation, setup, and checkpoint writing; future pauses and machine load can change the estimate.

Status freshness is based on the status file's modification time. The default training grace is 300 seconds; validation gets 1,200 seconds because it can run as one blocking evaluation. An old heartbeat is labeled “possibly unresponsive,” since a disk snapshot cannot distinguish a stalled process, process exit, or a long blocking operation. Tune these with `--stale-after-seconds` and `--validation-grace-seconds` for slower machines.

Recommendations are prompts for inspection grounded in measured telemetry. They do not change the run. Validation uses fixed seeds and may select the saved champion, so it is not an independent held-out transfer result. Outcomes describe this simulator and do not establish performance in the original game.
