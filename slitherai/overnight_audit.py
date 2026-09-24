"""Read-only, low-cost audit of a SlitherAI training run.

Only JSON telemetry and checkpoint filenames/metadata are read. Checkpoint
pickles and episode matrices are deliberately not loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BEHAVIOR_KEYS = (
    "food_gain", "boost_spent", "alive", "age", "kills", "border_death",
    "collision_death", "boost_fraction", "turn_degrees", "reward_growth",
    "reward_survival", "reward_kills", "reward_death",
)
ACTIVE_PHASES = {"training", "validating", "starting", "resuming"}
INACTIVE_PHASES = {"paused", "stopped", "finished", "complete", "completed", "interrupted", "error", "failed"}


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {type(exc).__name__}"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def _field(row: dict[str, Any], key: str) -> float | None:
    return _number(row.get(key))


def _history(run: Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = run / "history.jsonl"
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    issues.append(f"history.jsonl line {line_number} is incomplete or invalid")
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except FileNotFoundError:
        issues.append("history.jsonl is missing")
    except OSError as exc:
        issues.append(f"history.jsonl could not be read ({type(exc).__name__})")
    rows.sort(key=lambda row: _number(row.get("generation")) if _number(row.get("generation")) is not None else -1)
    return rows, issues


def _metrics(row: dict[str, Any]) -> dict[str, float | None]:
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    result: dict[str, float | None] = {}
    for key in ("best", "mean", "seconds", "species", "nodes", "connections"):
        result[key] = _field(row, key)
    result["selection_mean"] = _field(row, "mean")
    result["raw_episode_mean"] = _field(evaluation, "fitness")
    result["anchor_mean"] = _field(evaluation, "anchor_fitness")
    for key in BEHAVIOR_KEYS:
        result[key] = _field(evaluation, key)
    result["selection_minus_anchor"] = (
        result["selection_mean"] - result["anchor_mean"]
        if result["selection_mean"] is not None and result["anchor_mean"] is not None else None
    )
    return result


def _validation_rows(run: Path, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_generation: dict[int, dict[str, Any]] = {}
    for row in history:
        generation = _number(row.get("generation"))
        validation = row.get("validation")
        if generation is not None and isinstance(validation, dict):
            by_generation[int(generation)] = validation
    validation_dir = run / "validation"
    if validation_dir.is_dir():
        for path in validation_dir.glob("generation-*.json"):
            try:
                generation = int(path.stem.split("-")[-1])
            except ValueError:
                continue
            value, _ = _read_json(path)
            if isinstance(value, dict):
                by_generation[generation] = value
    rows = []
    for generation, value in sorted(by_generation.items()):
        metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else value
        rows.append({
            "generation": generation,
            "fitness": _round(_number(metrics.get("fitness", value.get("fitness")))),
            "standard_error": _round(_number(value.get("standard_error"))),
            "maps": _number(value.get("maps")),
            "seconds": _number(value.get("seconds")),
            "food_gain": _round(_number(metrics.get("food_gain"))),
            "alive": _round(_number(metrics.get("alive"))),
            "border_death": _round(_number(metrics.get("border_death"))),
            "collision_death": _round(_number(metrics.get("collision_death"))),
        })
    return rows


def _species_summary(run: Path, history: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    species, issue = _read_json(run / "species.json")
    warnings = []
    if issue and issue != "missing":
        warnings.append(f"species.json {issue}")
    species = species if isinstance(species, dict) else {}
    current = species.get("current") if isinstance(species.get("current"), dict) else {}
    latest = species.get("latest") if isinstance(species.get("latest"), dict) else {}
    parameters = species.get("parameters") if isinstance(species.get("parameters"), dict) else {}
    rows = current.get("rows") if isinstance(current.get("rows"), list) else []
    population = _number(current.get("population"))
    tiny_limit = max(2, math.ceil((population or 0) * 0.01))
    sizes = [_number(row.get("size")) for row in rows if isinstance(row, dict)]
    sizes = [size for size in sizes if size is not None]
    count = _number(current.get("count"))
    target_min = _number(parameters.get("target_min"))
    target_max = _number(parameters.get("target_max"))
    trail = []
    for row in history[-10:]:
        value = _number(row.get("species"))
        if value is not None:
            trail.append({"generation": int(_number(row.get("generation")) or 0), "count": value})
    return ({
        "current_generation": _number(current.get("generation")),
        "population": population,
        "count": count,
        "effective_diversity": _round(_number(current.get("effective"))),
        "largest_share": _round(_number(current.get("largest_share"))),
        "compatibility_threshold": _number(parameters.get("compatibility_threshold")),
        "target_count": {"min": target_min, "max": target_max},
        "tiny_size_limit": tiny_limit,
        "tiny_species_count": sum(size <= tiny_limit for size in sizes),
        "tiny_species_population": sum(size for size in sizes if size <= tiny_limit),
        "latest_evaluated_generation": _number(latest.get("generation")),
        "latest_birth_ids": latest.get("new_ids", []) if isinstance(latest.get("new_ids", []), list) else [],
        "latest_removed_ids": latest.get("removed_ids", []) if isinstance(latest.get("removed_ids", []), list) else [],
        "latest_removed_for_stagnation": sum(1 for row in latest.get("rows", [])
                                                if isinstance(row, dict) and row.get("removed")),
        "count_history_last_10": trail,
        "parameters": {key: parameters.get(key) for key in (
            "survival_threshold", "max_stagnation", "species_elitism", "elitism", "target_min", "target_max")
            if key in parameters},
    }, warnings)


def _freshness(run: Path, status: dict[str, Any], now: float,
               stale_after_seconds: float, validation_grace_seconds: float) -> dict[str, Any]:
    path = run / "status.json"
    try:
        age = max(0.0, now - path.stat().st_mtime)
    except OSError:
        age = None
    phase = str(status.get("phase", "unknown")).lower()
    if phase in INACTIVE_PHASES:
        state = "inactive"
        note = f"Run reports phase '{phase}'."
    elif phase in ACTIVE_PHASES:
        grace = validation_grace_seconds if phase == "validating" else stale_after_seconds
        if age is None:
            state = "unknown"
            note = "No readable status heartbeat is available."
        elif age <= grace:
            state = "recent_heartbeat"
            note = "The status file was updated recently; a long generation alone is not evidence of a stall."
        else:
            state = "possibly_unresponsive"
            note = "The status heartbeat is old; a stall, process exit, or long blocking step is possible."
    else:
        state = "unknown"
        note = "Run phase is missing or unrecognized; heartbeat age cannot establish process health."
    progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
    completed = _number(progress.get("completed_episodes"))
    total = _number(progress.get("total_episodes"))
    return {
        "state": state,
        "phase": phase,
        "status_age_seconds": _round(age, 1),
        "stale_after_seconds": stale_after_seconds,
        "validation_grace_seconds": validation_grace_seconds,
        "generation": _number(status.get("generation")),
        "progress": {
            "game": _number(progress.get("game")), "games": _number(progress.get("games")),
            "batch": _number(progress.get("batch")), "batches": _number(progress.get("batches")),
            "completed_episodes": completed, "total_episodes": total,
            "fraction": _round(completed / total, 4) if completed is not None and total else None,
            "episode_seconds": _number(status.get("episode_seconds")),
            "agent_steps_per_second": _number(status.get("agent_steps_per_second")),
        },
        "note": note,
    }


def _wall_budget(run: Path, history: list[dict[str, Any]], status: dict[str, Any],
                 settings: dict[str, Any], deadline: datetime | None, now: float) -> dict[str, Any]:
    durations = [_number(row.get("seconds")) for row in history[-5:]]
    durations = [value for value in durations if value is not None and value > 0]
    median = statistics.median(durations) if durations else None
    generation_ids = {int(value) for row in history
                      if (value := _number(row.get("generation"))) is not None}
    completed = len(generation_ids)
    latest_generation = max(generation_ids, default=-1)
    target = _number(status.get("target_generation"))
    target_source = "status.target_generation"
    if target is None:
        setting_generations = _number(settings.get("generations"))
        target = setting_generations
        target_source = "settings.generations (fallback; resume target may differ)"
    remaining = max(0, int(target) - (latest_generation + 1)) if target is not None else None
    estimate = median * remaining if median is not None and remaining is not None else None
    elapsed = _number(status.get("elapsed_wall"))
    budget_left = None
    margin = None
    fits = None
    if deadline is not None:
        budget_left = max(0.0, deadline.timestamp() - now)
        if estimate is not None:
            margin = budget_left - estimate
            fits = margin >= 0
    return {
        "completed_generations": completed,
        "latest_complete_generation": latest_generation if latest_generation >= 0 else None,
        "target_generation": target,
        "target_source": target_source,
        "remaining_generations_estimate": remaining,
        "recent_generation_seconds": [_round(value, 1) for value in durations],
        "recent_generation_seconds_median": _round(median, 1),
        "training_only_generations_per_hour": _round(3600 / median, 2) if median else None,
        "estimated_remaining_training_hours": _round(estimate / 3600, 2) if estimate is not None else None,
        "elapsed_wall_seconds_reported": _round(elapsed, 1),
        "deadline_utc": deadline.astimezone(timezone.utc).isoformat() if deadline else None,
        "deadline_budget_seconds": _round(budget_left, 1),
        "deadline_margin_seconds_estimate": _round(margin, 1),
        "training_estimate_fits_deadline": fits,
        "estimate_limit": "Uses the median of up to five completed history seconds fields. Pauses during those evaluations are included; validation, setup and checkpoint time are omitted, and future pauses or machine load can change the pace.",
    }


def _recommendations(history: list[dict[str, Any]], validation: list[dict[str, Any]],
                     species: dict[str, Any], status: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if history:
        latest = _metrics(history[-1])
        border, collision = latest.get("border_death"), latest.get("collision_death")
        if border is not None and collision is not None and border > collision and border >= 0.4:
            out.append({"priority": "high", "basis": f"Latest cohort border deaths {border:.1%} exceed collision deaths {collision:.1%}.",
                        "action": "Review boundary sensing and turn decisions in recorded episodes before changing selection settings."})
        food, spent = latest.get("food_gain"), latest.get("boost_spent")
        if food is not None and spent is not None and spent >= food and spent > 0:
            out.append({"priority": "medium", "basis": f"Latest cohort boost spend ({spent:.2f}) is at least food gain ({food:.2f}).",
                        "action": "Inspect boost timing and net growth alongside survival before drawing conclusions from boost frequency."})
        gap = latest.get("selection_minus_anchor")
        if gap is not None and abs(gap) >= 2:
            out.append({"priority": "medium", "basis": f"Latest population selection mean differs from fixed-anchor mean by {gap:+.2f} points.",
                        "action": "Check whether the rotating scenario is steering selection by comparing per-genome anchor and selection scores."})
    if validation:
        latest_validation = validation[-1]
        error = latest_validation.get("standard_error")
        if error is not None and error >= 2:
            out.append({"priority": "medium", "basis": f"Latest fixed validation standard error is {error:.2f} points.",
                        "action": "Treat small validation changes as uncertain; seek a separate held-out seed set before claiming transfer."})
    count = species.get("count")
    target = species.get("target_count", {})
    minimum, maximum = target.get("min"), target.get("max")
    trail = species.get("count_history_last_10", [])
    if count is not None and minimum is not None and maximum is not None:
        recent_counts = [row["count"] for row in trail[-3:]]
        if len(recent_counts) >= 3 and all(value < minimum or value > maximum for value in recent_counts):
            out.append({"priority": "low", "basis": f"Species count stayed outside the configured {minimum:g}–{maximum:g} range for the last three completed generations.",
                        "action": "Review compatibility-threshold response and current diversity trend before adjusting speciation targets."})
    if status.get("phase") in ACTIVE_PHASES and status.get("_heartbeat_state") == "possibly_unresponsive":
        out.append({"priority": "high", "basis": "The run reports an active phase but its status heartbeat exceeded the configured grace period.",
                    "action": "Check the local process and GPU activity; the file snapshot cannot distinguish a stall from process exit."})
    if not out:
        out.append({"priority": "low", "basis": "No configured audit trigger fired from the available telemetry.",
                    "action": "Continue monitoring completed generations and periodic fixed validation."})
    return out


def _parse_deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--deadline must include a timezone, for example 2026-09-25T08:00:00+02:00")
    return parsed


def build_audit(run: str | Path, *, now: float | None = None,
                stale_after_seconds: float = 300,
                validation_grace_seconds: float = 1200,
                deadline: datetime | None = None) -> dict[str, Any]:
    """Create a compact audit object from one run directory."""
    run_path = Path(run)
    now = time.time() if now is None else now
    history, warnings = _history(run_path)
    status_value, status_issue = _read_json(run_path / "status.json")
    settings_value, settings_issue = _read_json(run_path / "settings.json")
    status = status_value if isinstance(status_value, dict) else {}
    settings = settings_value if isinstance(settings_value, dict) else {}
    for name, issue in (("status.json", status_issue), ("settings.json", settings_issue)):
        if issue and issue != "missing":
            warnings.append(f"{name} {issue}")
    freshness = _freshness(run_path, status, now, stale_after_seconds, validation_grace_seconds)
    status["_heartbeat_state"] = freshness["state"]
    validation = _validation_rows(run_path, history)
    best_validation_value, best_issue = _read_json(run_path / "best-validation.json")
    if best_issue and best_issue != "missing":
        warnings.append(f"best-validation.json {best_issue}")
    best_validation = best_validation_value if isinstance(best_validation_value, dict) else None
    baseline, baseline_issue = _read_json(run_path / "baselines.json")
    if baseline_issue and baseline_issue != "missing":
        warnings.append(f"baselines.json {baseline_issue}")
    species, species_warnings = _species_summary(run_path, history)
    warnings.extend(species_warnings)
    latest = _metrics(history[-1]) if history else {}
    preceding = [_metrics(row) for row in history[-10:-5]]
    recent = [_metrics(row) for row in history[-5:]]
    behavior_windows = {}
    for key in ("selection_mean", "anchor_mean", "selection_minus_anchor", *BEHAVIOR_KEYS):
        old = [_field(row, key) for row in preceding]
        new = [_field(row, key) for row in recent]
        behavior_windows[key] = {"previous_up_to_5_mean": _round(_mean([x for x in old if x is not None])),
                                 "latest_up_to_5_mean": _round(_mean([x for x in new if x is not None]))}
    generation_durations = [_number(row.get("seconds")) for row in history[-5:]]
    generation_durations = [x for x in generation_durations if x is not None]
    checkpoints = []
    for path in run_path.glob("checkpoint-*"):
        try:
            generation = int(path.name.removeprefix("checkpoint-"))
            stat = path.stat()
        except (ValueError, OSError):
            continue
        checkpoints.append({"generation": generation, "size_bytes": stat.st_size,
                            "modified_age_seconds": _round(max(0, now - stat.st_mtime), 1)})
    checkpoints.sort(key=lambda row: row["generation"])
    budget = _wall_budget(run_path, history, status, settings, deadline, now)
    recommendations = _recommendations(history, validation, species, status)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "run": str(run_path.resolve()),
        "freshness": freshness,
        "progress": {
            "history_generations": len(history),
            "history_generation_range": [int(_number(history[0].get("generation")) or 0),
                                         int(_number(history[-1].get("generation")) or 0)] if history else None,
            "latest_generation": latest,
            "last_5_vs_previous_5": behavior_windows,
            "recent_generation_seconds_median": _round(statistics.median(generation_durations), 1) if generation_durations else None,
        },
        "validation": {
            "records": validation,
            "latest": validation[-1] if validation else None,
            "best_validation_file": best_validation,
            "baselines": baseline if isinstance(baseline, dict) else None,
            "comparison_note": "Validation uses fixed seeds for model selection; it is not an independent held-out test.",
        },
        "species": species,
        "wall_budget": budget,
        "checkpoint_metadata": {"count": len(checkpoints), "latest": checkpoints[-1] if checkpoints else None},
        "recommendations": recommendations,
        "uncertainty_and_limits": [
            "Scores are simulator outcomes; they do not establish transfer to the original game.",
            "Selection-versus-anchor values compare population means; they do not identify which genomes benefited from the rotating scenario.",
            "Validation reuses fixed seeds and may have been used to select the saved champion; use separate seeds for an independent claim.",
            "Generation-time estimates use up to five completed history seconds values; they include pauses during those evaluations but omit validation, setup and checkpoint writing. Future pauses and machine load can change the pace.",
            "Old status heartbeats are only evidence of possible unresponsiveness; this offline audit does not inspect processes or GPU activity.",
        ],
        "warnings": warnings,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    progress = audit["progress"]
    freshness = audit["freshness"]
    budget = audit["wall_budget"]
    validation = audit["validation"]
    species = audit["species"]
    latest = progress["latest_generation"]
    val = validation.get("latest")
    lines = [
        "# Overnight run audit",
        "",
        f"Run: `{audit['run']}`",
        f"Snapshot: {audit['generated_at_utc']}",
        "",
        "## Run state",
        "",
        f"- Phase: **{freshness['phase']}**; freshness: **{freshness['state']}** ({freshness['note']})",
        f"- Status heartbeat age: {freshness['status_age_seconds']} seconds",
        f"- Last complete generation: {budget['latest_complete_generation']}; target: {budget['target_generation']} ({budget['target_source']})",
        f"- In-flight progress: {freshness['progress']['completed_episodes']}/{freshness['progress']['total_episodes']} episodes; game {freshness['progress']['game']}/{freshness['progress']['games']}, batch {freshness['progress']['batch']}/{freshness['progress']['batches']}",
        "",
        "## Pace and deadline",
        "",
        f"- Recent generation median: {budget['recent_generation_seconds_median']} s ({budget['training_only_generations_per_hour']} generations/hour)",
        f"- Estimated remaining training: {budget['estimated_remaining_training_hours']} h for {budget['remaining_generations_estimate']} generations",
        f"- Deadline: {budget['deadline_utc'] or 'not supplied'}; estimated margin: {budget['deadline_margin_seconds_estimate']} s",
        f"- Timing basis: {budget['estimate_limit']}",
        "",
        "## Fitness and behavior",
        "",
        f"- Latest selection best/mean: {latest.get('best')} / {latest.get('selection_mean')}; fixed-anchor population mean: {latest.get('anchor_mean')}; selection-minus-anchor: {latest.get('selection_minus_anchor')}; raw episode outcome mean: {latest.get('raw_episode_mean')}.",
    ]
    behavior_labels = [("food_gain", "Food gain"), ("boost_spent", "Boost spend"), ("alive", "Survival fraction"),
                       ("border_death", "Border deaths"), ("collision_death", "Collision deaths"),
                       ("boost_fraction", "Boost decision fraction"), ("turn_degrees", "Direction change per decision (degrees)")]
    for key, label in behavior_labels:
        value = latest.get(key)
        if value is not None:
            lines.append(f"- {label}: {value:.3f} latest generation; last-five mean {progress['last_5_vs_previous_5'][key]['latest_up_to_5_mean']}.")
    lines.extend(["", "## Fixed validation", ""])
    if val:
        lines.append(f"- Latest generation {val['generation']}: fitness {val['fitness']}, standard error {val['standard_error']}, maps {val['maps']}; best saved validation: {validation.get('best_validation_file', {}).get('fitness') if validation.get('best_validation_file') else 'unavailable'}.")
    else:
        lines.append("- No fixed-validation record was found.")
    lines.extend([f"- {validation['comparison_note']}", "", "## Species", ""])
    lines.append(f"- Current count/effective diversity: {species['count']} / {species['effective_diversity']}; largest species share: {species['largest_share']}.")
    lines.append(f"- Compatibility threshold: {species['compatibility_threshold']}; target count: {species['target_count']['min']}–{species['target_count']['max']}.")
    lines.append(f"- Tiny species (size ≤ {species['tiny_size_limit']}): {species['tiny_species_count']} containing {species['tiny_species_population']} genomes; latest births/removals: {species['latest_birth_ids']} / {species['latest_removed_ids']}.")
    lines.extend(["", "## Suggested checks", ""])
    for item in audit["recommendations"]:
        lines.append(f"- **{item['priority']}** {item['basis']} {item['action']}")
    lines.extend(["", "## Limits", ""])
    for item in audit["uncertainty_and_limits"]:
        lines.append(f"- {item}")
    if audit["warnings"]:
        lines.extend(["", "## Data warnings", ""])
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="training run directory")
    parser.add_argument("--out", help="output directory (default: RUN/analysis/overnight)")
    parser.add_argument("--deadline", help="timezone-aware ISO timestamp, e.g. 2026-09-25T08:00:00+02:00")
    parser.add_argument("--stale-after-seconds", type=float, default=300,
                        help="training status age after which the run is marked possibly unresponsive")
    parser.add_argument("--validation-grace-seconds", type=float, default=1200,
                        help="longer heartbeat grace during fixed validation")
    args = parser.parse_args(argv)
    try:
        deadline = _parse_deadline(args.deadline)
        if args.stale_after_seconds <= 0 or args.validation_grace_seconds <= 0:
            raise ValueError("heartbeat grace periods must be positive")
        run = Path(args.run)
        if not run.is_dir():
            raise ValueError(f"run directory does not exist: {run}")
        audit = build_audit(run, stale_after_seconds=args.stale_after_seconds,
                            validation_grace_seconds=args.validation_grace_seconds,
                            deadline=deadline)
        output = Path(args.out) if args.out else run / "analysis" / "overnight"
        output.mkdir(parents=True, exist_ok=True)
        (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output / "audit.md").write_text(render_markdown(audit), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"json": str(output / "audit.json"), "markdown": str(output / "audit.md"),
                      "freshness": audit["freshness"]["state"],
                      "latest_generation": audit["wall_budget"]["latest_complete_generation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
