"""Read-only calibration summary for exported SlitherAI browser samples."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimConfig
from .schema import ANGLES, CHANNELS, INPUTS, contract, record_inputs, record_outputs


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return {"n": 0, "min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None}
    quantiles = np.quantile(clean, [.1, .5, .9, .95])
    return {"n": int(clean.size), "min": float(clean.min()),
            "p10": float(quantiles[0]), "p50": float(quantiles[1]),
            "p90": float(quantiles[2]), "p95": float(quantiles[3]), "max": float(clean.max())}


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_delta(previous: dict[str, Any], current: dict[str, Any]) -> tuple[float | None, str | None]:
    # Never subtract browser performance time from wall-clock time if a field is
    # missing at one endpoint.
    for key, source in (("t", "performance.now"), ("wallTimeMs", "wallTimeMs")):
        before, after = _finite(previous.get(key)), _finite(current.get(key))
        if before is not None and after is not None:
            return after-before, source
    return None, None


def _read_sim_config(settings: Path | None) -> SimConfig:
    if settings is None:
        return SimConfig().validate()
    try:
        payload = json.loads(settings.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read simulator settings: {type(exc).__name__}") from exc
    values = payload.get("config", payload) if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        raise ValueError("Simulator settings must be a config object or a run settings.json")
    return SimConfig.from_dict(values).validate()


def _iter_samples(path: Path, counts: dict[str, int], session_holder: dict[str, Any]):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            counts["lines"] += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                counts["other_records"] += 1
            elif record.get("type") == "session":
                session_holder["session"] = record
            elif record.get("type") == "sample":
                counts["samples"] += 1
                yield record
            else:
                counts["other_records"] += 1


def _local_reference(config: SimConfig, observed_arena_radius: dict[str, Any]) -> dict[str, Any]:
    radius = 14.5 * (1 + config.initial_mass / 300) ** .4
    size_factor = 1 + (radius / 14.5 - 1) * .1
    local_arena = config.arena_radius
    view_zoom = math.sqrt(radius / 14.5)
    nominal_food_density = config.foods / (math.pi * local_arena ** 2)
    median_source_arena = observed_arena_radius.get("p50")
    return {
        "config": dataclasses.asdict(config),
        "initial_nominal_body_radius": radius,
        "initial_mass_radius_range": [
            14.5 * (1 + config.initial_mass * .8 / 300) ** .4,
            14.5 * (1 + config.initial_mass * 1.2 / 300) ** .4,
        ],
        "nominal_base_speed_after_size_factor": config.base_speed * size_factor,
        "nominal_boost_speed_after_size_factor": config.boost_speed * size_factor,
        "nominal_boost_to_base_speed_ratio": config.boost_speed / config.base_speed,
        "raw_speed_scale_world_units_per_source_raw_unit": config.raw_speed_scale,
        "nominal_initial_turn_cap_radians_per_second": config.turn_rate * math.sqrt(14.5 / radius),
        "boost_cost_mass_per_action": config.boost_cost * config.dt,
        "boost_cost_mass_per_second_if_continuous": config.boost_cost,
        "nominal_view_full_width_at_initial_radius": 2 * config.view_half_width * view_zoom,
        "nominal_view_full_height_at_initial_radius": 2 * config.view_half_height * view_zoom,
        "initial_ambient_food_per_arena": config.foods,
        "initial_mobile_prey_within_food_slots": config.preys,
        "initial_nominal_ambient_food_density_per_world_unit2": nominal_food_density,
        "arena_radius_low_high": [local_arena * (1-config.arena_variation), local_arena * (1+config.arena_variation)],
        "source_to_local_nominal_arena_radius_ratio": median_source_arena / local_arena if median_source_arena else None,
    }


def analyze_export(source: Path, config: SimConfig | None = None) -> dict[str, Any]:
    source = source.resolve()
    config = (config or SimConfig()).validate()
    record_counts = {"lines": 0, "invalid_json": 0, "samples": 0, "other_records": 0}
    session_holder: dict[str, Any] = {}
    speed_raw = {"boost_requested": [], "boost_not_requested": []}
    raw_speed_feature = {key: [] for key in speed_raw}
    radius, scale_raw, segment_count, arena_radius = [], [], [], []
    ranges, viewport_width, viewport_height, width_per_size_zoom, height_per_size_zoom = [], [], [], [], []
    visible_food_density, visible_food_count, visible_prey_count, server_players, visible_enemies = [], [], [], [], []
    action_boost, action_steer, action_wanted_error = [], [], []
    angular_velocity, high_error_turn, high_error_local_cap_ratio = [], [], []
    movement_speed_raw_ratio = {key: [] for key in speed_raw}
    movement_world_speed = {key: [] for key in speed_raw}
    interval_seconds, interval_speed_eligible = [], 0
    contract_samples, contract_failures, angle_mismatches, ray_count_histogram = 0, 0, 0, {}
    channel_nonzero = [0] * len(CHANNELS)
    channel_sum = [0.0] * len(CHANNELS)
    ray_total = censored_rays = return_counts = 0
    return_kinds: dict[str, int] = {}
    output_boost, output_heading_turns = [], []
    alignment_errors, alignment_checks = [], 0
    previous: dict[str, Any] | None = None
    bad_intervals = 0
    scale_consistency = []
    first_performance_time = last_performance_time = None
    first_wall_time = last_wall_time = None
    performance_time_count = wall_time_count = 0
    interval_clock_counts: dict[str, int] = {}
    sample_count = 0

    for sample in _iter_samples(source, record_counts, session_holder):
        sample_count += 1
        performance_time, wall_time = _finite(sample.get("t")), _finite(sample.get("wallTimeMs"))
        performance_time_count += int(performance_time is not None)
        wall_time_count += int(wall_time is not None)
        if sample_count == 1:
            first_performance_time, first_wall_time = performance_time, wall_time
        if performance_time is not None: last_performance_time = performance_time
        if wall_time is not None: last_wall_time = wall_time
        lidar = sample.get("lidar") if isinstance(sample.get("lidar"), dict) else {}
        player = lidar.get("player") if isinstance(lidar.get("player"), dict) else {}
        rays = lidar.get("rays") if isinstance(lidar.get("rays"), list) else []
        if rays:
            ray_count_histogram[str(len(rays))] = ray_count_histogram.get(str(len(rays)), 0) + 1
            ray_total += len(rays)
            for index, ray in enumerate(rays):
                if not isinstance(ray, dict):
                    angle_mismatches += 1
                    continue
                if index >= len(ANGLES) or _finite(ray.get("relativeAngle")) is None or abs(float(ray["relativeAngle"]) - float(ANGLES[index])) > 1e-5:
                    angle_mismatches += 1
                ray_range = _finite(ray.get("range"))
                if ray_range is not None:
                    ranges.append(ray_range)
                censored_rays += int(bool(ray.get("censored")))
                returns = ray.get("returns") if isinstance(ray.get("returns"), list) else []
                return_counts += len(returns)
                for hit in returns:
                    if isinstance(hit, dict):
                        kind = str(hit.get("kind", "unknown"))
                        return_kinds[kind] = return_kinds.get(kind, 0) + 1
        try:
            encoded = record_inputs(sample)
            if encoded.size != INPUTS:
                raise ValueError("wrong observation length")
            channels = encoded[:len(ANGLES)*len(CHANNELS)].reshape(len(ANGLES), len(CHANNELS))
            for index in range(len(CHANNELS)):
                channel_nonzero[index] += int(np.count_nonzero(channels[:, index]))
                channel_sum[index] += float(channels[:, index].sum())
            contract_samples += 1
        except (KeyError, TypeError, ValueError, IndexError):
            contract_failures += 1

        action = sample.get("action") if isinstance(sample.get("action"), dict) else {}
        requested_boost = bool(action.get("boost"))
        group = "boost_requested" if requested_boost else "boost_not_requested"
        action_boost.append(float(requested_boost))
        p_speed = _finite(player.get("speedRaw"))
        if p_speed is not None:
            speed_raw[group].append(p_speed)
            raw_speed_feature[group].append(min(p_speed / 12.0, 1.0))
        p_radius = _finite(player.get("bodyRadiusEstimate"))
        p_scale = _finite(player.get("scaleRaw"))
        p_segments = _finite(player.get("segmentCount"))
        if p_radius is not None: radius.append(p_radius)
        if p_scale is not None: scale_raw.append(p_scale)
        if p_segments is not None: segment_count.append(p_segments)
        if p_scale is not None and p_radius is not None:
            scale_consistency.append(abs(p_radius - min(80., max(6., p_scale*14.5))))
        telem = lidar.get("telemetry") if isinstance(lidar.get("telemetry"), dict) else {}
        arena_value = _finite(telem.get("arenaRadiusEstimate"))
        if arena_value is not None: arena_radius.append(arena_value)

        p_heading, wanted = _finite(player.get("heading")), _finite(player.get("wantedHeading"))
        steering = _finite(action.get("steeringAngle"))
        if steering is not None: action_steer.append(steering)
        try:
            outputs = record_outputs(sample)
            output_boost.append(float(outputs[0]))
            output_heading_turns.append(float(outputs[1]))
            if p_heading is not None and wanted is not None:
                alignment_errors.append(abs(_wrap(float(outputs[1] * math.tau) - wanted)))
                alignment_checks += 1
        except (KeyError, TypeError, ValueError):
            pass
        if p_heading is not None and wanted is not None:
            action_wanted_error.append(abs(_wrap(wanted-p_heading)))

        camera = lidar.get("camera") if isinstance(lidar.get("camera"), dict) else {}
        rect = camera.get("visibleRectCanvas") if isinstance(camera.get("visibleRectCanvas"), dict) else {}
        zoom = _finite(camera.get("scale"))
        left, top = _finite(rect.get("left")), _finite(rect.get("top"))
        right, bottom = _finite(rect.get("right")), _finite(rect.get("bottom"))
        if zoom is not None and zoom > 0 and None not in (left, top, right, bottom) and right > left and bottom > top:
            width, height = (right-left)/zoom, (bottom-top)/zoom
            viewport_width.append(width)
            viewport_height.append(height)
            if p_radius is not None and p_radius > 0:
                size_zoom = math.sqrt(p_radius/14.5)
                width_per_size_zoom.append(width/size_zoom)
                height_per_size_zoom.append(height/size_zoom)
            area = width*height
            items = lidar.get("visibleFood") if isinstance(lidar.get("visibleFood"), list) else []
            food_n = sum(1 for item in items if isinstance(item, dict) and item.get("source") == "food")
            prey_n = sum(1 for item in items if isinstance(item, dict) and item.get("source") == "prey")
            visible_food_count.append(float(food_n))
            visible_prey_count.append(float(prey_n))
            if area > 0:
                visible_food_density.append(food_n / area * 1_000_000)
        counts = lidar.get("counts") if isinstance(lidar.get("counts"), dict) else {}
        for value, target in ((telem.get("playersOnServer"), server_players), (counts.get("enemies"), visible_enemies)):
            number = _finite(value)
            if number is not None: target.append(number)

        if previous is not None:
            elapsed_ms, interval_clock = _timestamp_delta(previous, sample)
            if interval_clock is not None:
                interval_clock_counts[interval_clock] = interval_clock_counts.get(interval_clock, 0) + 1
            prev_lidar = previous.get("lidar") if isinstance(previous.get("lidar"), dict) else {}
            prev_player = prev_lidar.get("player") if isinstance(prev_lidar.get("player"), dict) else {}
            dt = elapsed_ms/1000 if elapsed_ms is not None else None
            if dt is not None and dt > 0:
                interval_seconds.append(dt)
                prev_action = previous.get("action") if isinstance(previous.get("action"), dict) else {}
                prev_group = "boost_requested" if bool(prev_action.get("boost")) else "boost_not_requested"
                prev_speed = _finite(prev_player.get("speedRaw"))
                prev_radius = _finite(prev_player.get("bodyRadiusEstimate"))
                prev_heading = _finite(prev_player.get("heading"))
                prev_wanted = _finite(prev_player.get("wantedHeading"))
                prev_x, prev_y = _finite(prev_player.get("x")), _finite(prev_player.get("y"))
                cur_x, cur_y = _finite(player.get("x")), _finite(player.get("y"))
                if dt <= .3 and prev_heading is not None and p_heading is not None:
                    turn_rate = abs(_wrap(p_heading-prev_heading))/dt
                    angular_velocity.append(turn_rate)
                    if prev_wanted is not None and abs(_wrap(prev_wanted-prev_heading)) >= math.radians(30):
                        high_error_turn.append(turn_rate)
                        if prev_radius is not None and prev_radius > 0:
                            cap = 2.8 * math.sqrt(14.5/prev_radius)
                            high_error_local_cap_ratio.append(turn_rate/cap)
                stable_boost = prev_group == group
                stable_speed = prev_speed is not None and p_speed is not None and prev_speed > 0 and abs(p_speed-prev_speed) <= .1*prev_speed
                straight = prev_heading is not None and p_heading is not None and abs(_wrap(p_heading-prev_heading)) <= math.radians(15)
                if dt <= .3 and stable_boost and stable_speed and straight and prev_speed and None not in (prev_x, prev_y, cur_x, cur_y):
                    distance = math.hypot(cur_x-prev_x, cur_y-prev_y)
                    world_speed = distance/dt
                    movement_world_speed[group].append(world_speed)
                    movement_speed_raw_ratio[group].append(world_speed/prev_speed)
                    interval_speed_eligible += 1
            else:
                bad_intervals += 1
        previous = sample

    radius_stats = _quantiles(radius)
    arena_stats = _quantiles(arena_radius)
    config = config.validate()
    local = _local_reference(config, arena_stats)
    if performance_time_count == sample_count and first_performance_time is not None and last_performance_time is not None:
        elapsed_sample_seconds = (last_performance_time-first_performance_time)/1000
        elapsed_clock = "performance.now"
    elif wall_time_count == sample_count and first_wall_time is not None and last_wall_time is not None:
        elapsed_sample_seconds = (last_wall_time-first_wall_time)/1000
        elapsed_clock = "wallTimeMs"
    else:
        elapsed_sample_seconds, elapsed_clock = None, None
    return {
        "format": "slitherai-source-fidelity-v1",
        "source": {"path": str(source), "sha256": _file_hash(source),
                   "bytes": source.stat().st_size, "session": {key: session_holder["session"].get(key) for key in (
                       "exportFormat", "schemaVersion", "extensionVersion", "startedAt", "endedAt",
                       "status", "endedReason", "samples", "finalScore", "finalRank")}
                   if session_holder.get("session") else None},
        "records": record_counts,
        "contract": {"expected": contract(), "valid_samples": contract_samples,
                     "failed_samples": contract_failures, "ray_count_histogram": ray_count_histogram,
                     "ray_angle_mismatches": angle_mismatches,
                     "mean_nonzero_ray_fraction_by_channel": {
                         name: channel_nonzero[i]/(contract_samples*len(ANGLES)) if contract_samples else None
                         for i, name in enumerate(CHANNELS)},
                     "mean_encoded_value_by_channel": {
                         name: channel_sum[i]/(contract_samples*len(ANGLES)) if contract_samples else None
                         for i, name in enumerate(CHANNELS)},
                     "range_units": "client world-coordinate units",
                     "visible_range": _quantiles(ranges),
                     "censored_ray_fraction": censored_rays/ray_total if ray_total else None,
                     "return_count": return_counts, "return_kinds": return_kinds},
        "timing": {"sample_count": sample_count, "elapsed_sample_seconds": elapsed_sample_seconds if sample_count > 1 else None,
                   "elapsed_clock": elapsed_clock, "interval_clock_counts": interval_clock_counts,
                   "interval_seconds": _quantiles(interval_seconds), "nonpositive_intervals": bad_intervals,
                   "movement_calibration_intervals": interval_speed_eligible},
        "actions": {"boost_requested_sample_fraction": statistics.mean(action_boost) if action_boost else None,
                    "boost_output_sample_fraction": statistics.mean(output_boost) if output_boost else None,
                    "steering_angle_radians": _quantiles(action_steer),
                    "wanted_heading_error_radians": _quantiles(action_wanted_error),
                    "encoded_target_vs_wanted_heading_error_radians": _quantiles(alignment_errors),
                    "target_alignment_checks": alignment_checks,
                    "direction_output_turns": _quantiles(output_heading_turns),
                    "semantics": "boost is held mouse/space input; steeringAngle is pointer target relative to heading; no boost-active or mass-cost field"},
        "speed": {"speedRaw_by_boost_request": {key: _quantiles(value) for key, value in speed_raw.items()},
                  "encoded_speed_feature_by_boost_request": {key: _quantiles(value) for key, value in raw_speed_feature.items()},
                  "observed_displacement_speed_world_units_per_second": {key: _quantiles(value) for key, value in movement_world_speed.items()},
                  "observed_world_speed_per_speedRaw": {key: _quantiles(value) for key, value in movement_speed_raw_ratio.items()},
                  "speed_interval_filter": "gap <=0.3s; boost request stable at both samples; speedRaw changes <=10%; heading changes <=15 degrees; previous speedRaw is denominator",
                  "interpretation": "position chord speed can underestimate path speed during turns/collisions and server lag; ratio is empirical, not a verified movement formula"},
        "turning": {"observed_absolute_heading_rate_radians_per_second": _quantiles(angular_velocity),
                    "rate_when_previous_heading_error_at_least_30_degrees": _quantiles(high_error_turn),
                    "observed_rate_over_sim_formula_cap": _quantiles(high_error_local_cap_ratio),
                    "sim_formula_comparison": "observed/(2.8*sqrt(14.5/source_bodyRadiusEstimate)); observed human turns are chosen rates, not a maximum-turn measurement"},
        "size": {"source_body_radius_estimate_units": radius_stats, "source_scaleRaw": _quantiles(scale_raw),
                 "source_segment_count": _quantiles(segment_count),
                 "radius_estimate_vs_clamped_14_5_times_scaleRaw_abs_error": _quantiles(scale_consistency),
                 "semantics": "radius estimate is the extension's community formula, not measured server collision radius; score/segments do not identify simulator mass"},
        "arena": {"extension_arena_radius_estimate_units": arena_stats,
                  "source_to_sim_radius_ratio": local["source_to_local_nominal_arena_radius_ratio"],
                  "source_formula": "0.98 * client grd; an estimate, not a verified lethal boundary"},
        "view_and_density": {"source_view_full_width_world_units": _quantiles(viewport_width),
                             "source_view_full_height_world_units": _quantiles(viewport_height),
                             "width_div_sqrt_radius_scale": _quantiles(width_per_size_zoom),
                             "height_div_sqrt_radius_scale": _quantiles(height_per_size_zoom),
                             "visible_food_per_million_viewport_world_units2": _quantiles(visible_food_density),
                             "visible_food_count": _quantiles(visible_food_count), "visible_prey_count": _quantiles(visible_prey_count),
                             "server_player_count": _quantiles(server_players), "visible_enemy_count": _quantiles(visible_enemies),
                             "density_limit": "viewport-clipped food and camera area; clustered food and camera motion prevent identifying global density"},
        "simulator_reference": local,
        "limitations": [
            "Human action boost is an input request, not proof that the game activated boost.",
            "The export has no mass, active boost state, or boost-spend counter; boost cost cannot be identified.",
            "Observed heading changes are selected human turns and sampled intermittently; they do not reveal the maximum turn rate.",
            "A score/segment/radius estimate does not identify source mass or calibrate the simulator's mass-growth curve.",
            "Food counts describe camera-visible objects, not the full arena density; opponents and food clustering are not controlled.",
            "Client world coordinates and clocks do not remove game-server lag, client interpolation, or camera/view uncertainty.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    observed = report["records"]
    lines = ["# Source-game fidelity calibration", "",
             f"Input: `{Path(report['source']['path']).name}`; SHA-256 `{report['source']['sha256']}`  ",
             f"Samples: {observed['samples']}; parsed invalid JSON lines: {observed['invalid_json']}; valid 530-input samples: {report['contract']['valid_samples']}", "",
             "Statistics are computed from browser timestamps and game coordinates. They are descriptive measurements, not proof that client fields equal server physics.", "",
             "## Measured quantities", "",
             "| Quantity | Observed export | Simulator reference | Interpretation |", "|---|---:|---:|---|"]
    def p50(item: dict[str, Any]) -> str:
        value = item.get("p50")
        return "—" if value is None else f"{value:.4g} (p10–p90 {item['p10']:.4g}–{item['p90']:.4g}; n={item['n']})"
    lines += [
        f"| Arena radius (world units) | {p50(report['arena']['extension_arena_radius_estimate_units'])} | {report['simulator_reference']['config']['arena_radius']:g} ± {report['simulator_reference']['config']['arena_variation']*100:g}% | Extension estimate is 0.98×client `grd`; client/source lethal edge is unverified. |",
        f"| Body radius estimate | {p50(report['size']['source_body_radius_estimate_units'])} | {report['simulator_reference']['initial_nominal_body_radius']:.3f} at configured initial mass | Source uses clamped 14.5×`sc`; this is not server collision radius or mass. |",
        f"| World displacement / `speedRaw` | boost request off: {p50(report['speed']['observed_world_speed_per_speedRaw']['boost_not_requested'])}; on: {p50(report['speed']['observed_world_speed_per_speedRaw']['boost_requested'])} | conversion scale {report['simulator_reference']['raw_speed_scale_world_units_per_source_raw_unit']:g} | Filtered straight, short intervals; chord displacement may underestimate speed. |",
        f"| Observed turn rate | {p50(report['turning']['observed_absolute_heading_rate_radians_per_second'])} rad/s | nominal initial cap {report['simulator_reference']['nominal_initial_turn_cap_radians_per_second']:.3f} rad/s | Observed choices do not identify the game maximum. |",
        f"| View full width / height | {p50(report['view_and_density']['source_view_full_width_world_units'])} / {p50(report['view_and_density']['source_view_full_height_world_units'])} | {report['simulator_reference']['nominal_view_full_width_at_initial_radius']:.1f} / {report['simulator_reference']['nominal_view_full_height_at_initial_radius']:.1f} | Source dimensions vary with camera, display size, zoom, and head offset. |",
        f"| Visible food per million view units² | {p50(report['view_and_density']['visible_food_per_million_viewport_world_units2'])} | initial arena average {report['simulator_reference']['initial_nominal_ambient_food_density_per_world_unit2']*1_000_000:.3f} per million | Viewport count is local and clustered; not global density. |",
        f"| Boost request in sampled frames | {report['actions']['boost_requested_sample_fraction'] if report['actions']['boost_requested_sample_fraction'] is not None else '—'} | local cost {report['simulator_reference']['boost_cost_mass_per_action']:g} mass/action; {report['simulator_reference']['boost_cost_mass_per_second_if_continuous']:g} mass/s | Source has no actual boost-active or mass-spent field; cost is unidentifiable. |",
    ]
    lines += ["", "## Sensor and action contract", "",
              f"Ray count by sample: `{json.dumps(report['contract']['ray_count_histogram'], sort_keys=True)}`; ray angle mismatches: {report['contract']['ray_angle_mismatches']}; valid observations: {report['contract']['valid_samples']}/{report['records']['samples']}.",
              "The extension v0.6.0 and simulator both define 87 ego-relative rays over ±166°, six channels per ray, eight globals, and two actions. The shared importer uses the same ray angles, `range/(range+1000)`, `1/(1+distance/100)`, food-size weighting, and absolute heading mapping. Exported boost is held mouse/space state; direction is cursor steering relative to current heading, converted to an absolute turn fraction. The simulator thresholds boost at 0.5 and moves toward an absolute target heading.", "",
              "Code differences that matter for transfer:", "",
              "- View range is camera-clipped in the extension and uses a virtual rectangle scaled by the simulator's body radius. Browser aspect ratio, viewport size, camera centering, and game zoom can change range values.",
              "- Enemy head/body sensors both add a 3-unit margin in each implementation, but the simulator's radii come from simulated mass while the extension estimates radius from client `sc`.",
              "- Own-body lidar differs: the extension inflates self capsules to `2×radius+3` and skips geometry near the head; the simulator uses self radius and omits capsules within its `3×radius` along-body cutoff. This is a real observation-semantic mismatch.",
              "- Both group prey with food, but food size, collision radius, position, and source availability differ; the extension receives client food `sz`, the simulator derives food size from its mass model.",
              "- Border sensor geometry is structurally similar (`arena radius − own radius − 3`), but the extension boundary is based on `0.98×grd`; the local arena's radius/center are configured independently.", "",
              "## Limits and reproduction", "",
              "The export does not contain mass or a reliable boost-spend counter. Public length/score and `sc` are not enough to infer mass without a verified source conversion. Non-boost intervals, high turn error, and timestamped head displacement can estimate a speed scale and describe chosen turns; they cannot establish maximum speed, boost cost, or maximum turn rate. Visible food is clipped to the viewport and affected by clusters and game population.", "",
              "To reproduce when the export is available:", "",
              "```powershell",
              ".venv\\Scripts\\python.exe -m slitherai.source_fidelity `",
              "  C:\\Users\\88mat\\Downloads\\slitherai-2026-09-23T15-09-52-803Z-b2039f8f.jsonl `",
              "  --settings runs\\20260924-163112-852097\\settings.json `",
              "  --out runs\\20260924-163112-852097\\analysis\\source-fidelity",
              "```", "",
              "The analyzer writes aggregate JSON and Markdown only; it does not persist individual samples or feed training. It accepts either an exported JSONL with a session header and sample rows or older rows with the same `sample` structure.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only calibration summary for a browser JSONL export")
    parser.add_argument("source", type=Path, help="Trusted local slitherai JSONL export")
    parser.add_argument("--settings", type=Path, help="Optional run settings.json or SimConfig JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output filename stem; creates .json and .md")
    args = parser.parse_args()
    report = analyze_export(args.source, _read_sim_config(args.settings))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.out.with_suffix('.json').resolve()),
                      "markdown": str(args.out.with_suffix('.md').resolve()),
                      "samples": report["records"]["samples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
