import json
import os
from datetime import datetime, timezone

from slitherai.overnight_audit import build_audit, render_markdown


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(path, *, status_age=20):
    history = [
        {"generation": 0, "best": 4.0, "mean": 1.0, "seconds": 100,
         "evaluation": {"fitness": -4.0, "anchor_fitness": 0.5, "food_gain": 3.0,
                        "boost_spent": 1.0, "alive": 0.2, "border_death": 0.6,
                        "collision_death": 0.2, "boost_fraction": 0.15, "turn_degrees": 8}},
        {"generation": 1, "best": 7.0, "mean": 2.0, "seconds": 120,
         "evaluation": {"fitness": 8.25, "anchor_fitness": -1.0, "food_gain": 2.0,
                        "boost_spent": 3.0, "alive": 0.3, "border_death": 0.7,
                        "collision_death": 0.1, "boost_fraction": 0.25, "turn_degrees": 9},
         "validation": {"fitness": 5.0, "standard_error": 2.5, "maps": 8,
                        "metrics": {"fitness": 5.0, "food_gain": 6.0, "alive": 0.4}}},
    ]
    path.mkdir(parents=True, exist_ok=True)
    (path / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history) + "\n", encoding="utf-8")
    write_json(path / "status.json", {
        "phase": "training", "generation": 2, "target_generation": 4, "elapsed_wall": 245,
        "progress": {"game": 2, "games": 5, "batch": 1, "batches": 4,
                     "completed_episodes": 300, "total_episodes": 1280},
    })
    os.utime(path / "status.json", (10_000 - status_age, 10_000 - status_age))
    write_json(path / "settings.json", {"generations": 4})
    write_json(path / "species.json", {
        "current": {"generation": 2, "population": 10, "count": 4, "effective": 3.5,
                    "largest_share": 0.4, "rows": [{"size": 5}, {"size": 2}, {"size": 2}, {"size": 1}]},
        "latest": {"generation": 1, "new_ids": [5], "removed_ids": [2], "rows": [{"removed": True}]},
        "parameters": {"compatibility_threshold": 1.7, "target_min": 3, "target_max": 8,
                       "survival_threshold": 0.3},
    })
    write_json(path / "best-validation.json", {"generation": 1, "fitness": 5.0})
    (path / "checkpoint-2").write_bytes(b"not a pickle and should only be stat'ed")


def test_builds_compact_audit_with_live_progress_validation_species_and_grounded_actions(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    deadline = datetime.fromtimestamp(10_100, tz=timezone.utc)

    audit = build_audit(run, now=10_000, stale_after_seconds=100, deadline=deadline)

    assert audit["freshness"]["state"] == "recent_heartbeat"
    assert audit["freshness"]["progress"]["fraction"] == round(300 / 1280, 4)
    assert audit["progress"]["latest_generation"]["selection_minus_anchor"] == 3.0
    assert audit["progress"]["latest_generation"]["selection_mean"] == 2.0
    assert audit["progress"]["latest_generation"]["raw_episode_mean"] == 8.25
    assert audit["validation"]["latest"]["fitness"] == 5.0
    assert audit["validation"]["latest"]["standard_error"] == 2.5
    assert audit["species"]["effective_diversity"] == 3.5
    assert audit["species"]["tiny_species_count"] == 3
    assert audit["species"]["latest_birth_ids"] == [5]
    assert audit["wall_budget"]["remaining_generations_estimate"] == 2
    assert audit["wall_budget"]["estimated_remaining_training_hours"] == 0.06
    assert audit["wall_budget"]["training_estimate_fits_deadline"] is False
    assert any("border deaths" in item["basis"] for item in audit["recommendations"])
    assert any("boost spend" in item["basis"] for item in audit["recommendations"])
    assert "not an independent held-out test" in audit["validation"]["comparison_note"]
    assert audit["checkpoint_metadata"]["latest"]["generation"] == 2
    assert "Latest fixed validation" in render_markdown(audit)


def test_old_active_heartbeat_is_uncertain_and_sparse_or_partial_files_are_tolerated(tmp_path):
    run = tmp_path / "sparse"
    run.mkdir()
    (run / "history.jsonl").write_text('{"generation": 0, "best": 1}\n{"generation":', encoding="utf-8")
    write_json(run / "status.json", {"phase": "training"})
    os.utime(run / "status.json", (9_000, 9_000))
    write_json(run / "settings.json", {"generations": 3})

    audit = build_audit(run, now=10_000, stale_after_seconds=300)

    assert audit["freshness"]["state"] == "possibly_unresponsive"
    assert "possible" in audit["freshness"]["note"]
    assert audit["validation"]["latest"] is None
    assert audit["species"]["count"] is None
    assert audit["wall_budget"]["remaining_generations_estimate"] == 2
    assert any("incomplete or invalid" in warning for warning in audit["warnings"])
    assert any("status heartbeat exceeded" in item["basis"] for item in audit["recommendations"])


def test_validation_phase_uses_its_longer_grace_period(tmp_path):
    run = tmp_path / "validating"
    run.mkdir()
    write_json(run / "status.json", {"phase": "validating"})
    os.utime(run / "status.json", (9_500, 9_500))

    audit = build_audit(run, now=10_000, stale_after_seconds=100, validation_grace_seconds=600)

    assert audit["freshness"]["state"] == "recent_heartbeat"
    assert not any(item["basis"].startswith("The run reports an active phase")
                   for item in audit["recommendations"])

    write_json(run / "status.json", {"phase": "completed"})
    completed = build_audit(run, now=10_000, stale_after_seconds=100)
    assert completed["freshness"]["state"] == "inactive"

    write_json(run / "status.json", {"phase": "interrupted"})
    interrupted = build_audit(run, now=10_000, stale_after_seconds=100)
    assert interrupted["freshness"]["state"] == "inactive"
