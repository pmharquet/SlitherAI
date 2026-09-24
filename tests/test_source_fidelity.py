import json

import pytest

from slitherai.config import SimConfig
from slitherai.schema import ANGLES
from slitherai.source_fidelity import analyze_export, render_markdown


def sample(index, timestamp, x):
    rays = [dict(relativeAngle=float(angle), range=500., censored=True, returns=[])
            for angle in ANGLES]
    rays[43]["returns"] = [dict(kind="food", distance=100., size=20.)]
    player = dict(x=x, y=0., heading=0., wantedHeading=0., speedRaw=5., scaleRaw=1.,
                  bodyRadiusEstimate=14.5, segmentCount=3, segmentFraction=.25)
    camera = dict(scale=1., visibleRectCanvas=dict(left=0., top=0., right=1200., bottom=1000.))
    return dict(type="sample", index=index, t=timestamp, wallTimeMs=1_000+timestamp,
                lidar=dict(player=player, rays=rays, camera=camera,
                           telemetry=dict(arenaRadiusEstimate=2400., playersOnServer=16),
                           counts=dict(enemies=3),
                           visibleFood=[dict(source="food", x=100., y=20.)]),
                action=dict(boost=False, steeringAngle=0.))


def test_synthetic_export_quantifies_time_speed_view_and_shared_sensor_contract(tmp_path):
    source = tmp_path / "source.jsonl"
    records = [dict(type="session", exportFormat="slitherai-jsonl-v1", samples=2,
                    extensionVersion="0.6.0", status="stopped"),
               sample(0, 0., 0.), sample(1, 100., 10.)]
    source.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    config = SimConfig(maps=1, worms=2, foods=8, preys=0, body_points=8, arena_radius=400)

    report = analyze_export(source, config)
    assert report["records"]["samples"] == 2
    assert report["contract"]["valid_samples"] == 2
    assert report["contract"]["ray_count_histogram"] == {"87": 2}
    assert report["contract"]["ray_angle_mismatches"] == 0
    assert report["contract"]["mean_nonzero_ray_fraction_by_channel"]["food_value"] > 0
    assert report["timing"]["elapsed_sample_seconds"] == pytest.approx(.1)
    ratio = report["speed"]["observed_world_speed_per_speedRaw"]["boost_not_requested"]
    assert ratio["p50"] == pytest.approx(20.)
    assert report["view_and_density"]["source_view_full_width_world_units"]["p50"] == pytest.approx(1200.)
    assert report["arena"]["extension_arena_radius_estimate_units"]["p50"] == pytest.approx(2400.)
    assert len(report["source"]["sha256"]) == 64
    markdown = render_markdown(report)
    assert "Own-body lidar differs" in markdown
    assert "unidentifiable" in markdown


def test_missing_optional_fields_and_corrupt_lines_are_reported_without_guessing(tmp_path):
    source = tmp_path / "partial.jsonl"
    row = sample(0, 0., 0.)
    del row["lidar"]["camera"]
    source.write_text("{bad json\n" + json.dumps(row) + "\n", encoding="utf-8")

    report = analyze_export(source)
    assert report["records"]["invalid_json"] == 1
    assert report["records"]["samples"] == 1
    assert report["contract"]["valid_samples"] == 1
    assert report["contract"]["failed_samples"] == 0
    assert report["view_and_density"]["source_view_full_width_world_units"]["n"] == 0
    assert report["speed"]["observed_world_speed_per_speedRaw"]["boost_not_requested"]["n"] == 0
