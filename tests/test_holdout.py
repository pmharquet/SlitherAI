import dataclasses
import json
import pickle
import random

import neat
import numpy as np
import pytest

from slitherai.config import SimConfig
from slitherai.evaluate_holdout import (
    _paired_delta,
    _runtime_provenance,
    _validate_genome_config,
    render_markdown,
    run_holdout,
)
from slitherai.network import load_config
from slitherai.protocol import protocol_settings
from slitherai.schema import VERSION, contract
from slitherai.train import AtomicCheckpointer


def make_run(path, reward_version="growth-v2"):
    config = SimConfig(maps=2, worms=4, foods=16, preys=0, body_points=8,
                       arena_radius=400, view_half_width=180, view_half_height=160,
                       reward_version=reward_version)
    path.mkdir()
    (path / "episodes").mkdir()
    (path / "settings.json").write_text(json.dumps({
        "config": dataclasses.asdict(config), "protocol": protocol_settings(), "seed": 7,
    }), encoding="utf-8")
    (path / "schema.json").write_text(json.dumps(contract()), encoding="utf-8")

    random.seed(30)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome_ids = list(population.population)
    for index, genome in enumerate(population.population.values()):
        genome.fitness = 1000. - index
    checkpoint = AtomicCheckpointer(1, filename_prefix=str(path / "checkpoint-"))
    checkpoint.save_checkpoint(population.config, population.population, population.species, 0)
    # Episode score intentionally disagrees with cached checkpoint fitness.
    scores = [1., 7., 2., 0.]
    (path / "episodes" / "generation-0000.json").write_text(json.dumps({
        "generation": 0, "genome_ids": genome_ids, "score": scores,
    }), encoding="utf-8")
    return config, neat_config, population, genome_ids


def test_holdout_reconstructs_generation_champion_and_replays_every_policy(tmp_path):
    run = tmp_path / "run"
    config, _, population, genome_ids = make_run(run)
    result = run_holdout(run, [0], [], seed=8231, maps=2, seconds=.2, device="cpu")

    assert [model["name"] for model in result["models"]] == [
        "initial", "candidate-generation-0", "heuristic", "circle",
    ]
    selected = result["models"][0]["source"]
    assert selected["kind"] == "checkpoint_episode_champion"
    assert selected["genome_id"] == genome_ids[1]
    assert selected["selection_score_used_only_to_select"] == 7.
    assert len(selected["genome_sha256"]) == 64
    assert len(selected["checkpoint_sha256"]) == 64
    assert result["comparison"]["config"]["maps"] == 2
    assert result["comparison"]["config"]["reward_version"] == config.reward_version
    assert len(result["comparison"]["config_sha256"]) == 64
    assert len(result["comparison"]["protocol_sha256"]) == 64
    assert len(result["comparison"]["reward_version_sha256"]) == 64
    assert result["comparison"]["all_policies_freshly_rescored"] is True
    runtime = result["runtime"]
    assert runtime["python_version"] and runtime["torch_version"] and runtime["neat_python_version"]
    assert set(runtime["code_sha256"]) == {
        "slitherai/config.py", "slitherai/evaluation.py", "slitherai/evaluate_holdout.py",
        "slitherai/network.py", "slitherai/protocol.py", "slitherai/rewards.py",
        "slitherai/schema.py", "slitherai/sim.py",
    }
    assert all(len(digest) == 64 for digest in runtime["code_sha256"].values())
    for model in result["models"]:
        for values in model["metrics"].values():
            assert len(values["per_map"]) == 2
    identical = next(item for item in result["paired_comparisons"]
                     if item["challenger"] == "candidate-generation-0" and item["comparator"] == "initial")
    assert identical["metrics"]["fitness"]["mean_difference"] == pytest.approx(0.)
    assert identical["metrics"]["fitness"]["bootstrap_95_percentile_interval"] == [0., 0.]
    report = render_markdown(result)
    assert "Paired fitness differences" in report and "Genome SHA-256" in report


def test_old_reward_requires_explicit_full_comparison_config_and_rescores_all(tmp_path):
    run = tmp_path / "old-run"
    config, _, _, _ = make_run(run, reward_version="legacy-v1")
    with pytest.raises(ValueError, match="comparison-config"):
        run_holdout(run, [0], [], seed=1902, maps=1, seconds=.1, device="cpu")

    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"config": dataclasses.asdict(config),
                                      "protocol": protocol_settings()}), encoding="utf-8")
    result = run_holdout(run, [0], [], seed=1902, maps=1, seconds=.1,
                         device="cpu", comparison_path=comparison)
    assert result["comparison"]["reward_version"] == "legacy-v1"
    assert result["comparison"]["all_policies_freshly_rescored"] is True
    assert len(result["models"]) == 4


def test_paired_summary_is_deterministic_and_reports_within_suite_error():
    a, b = [4., 8., 3., 9.], [1., 4., 5., 8.]
    first = _paired_delta(a, b, 741, "candidate:baseline:fitness")
    second = _paired_delta(a, b, 741, "candidate:baseline:fitness")
    assert first == second
    assert first["mean_difference"] == pytest.approx(1.5)
    assert first["paired_standard_error"] > 0
    assert first["bootstrap_95_percentile_interval"][0] <= 1.5
    assert first["bootstrap_95_percentile_interval"][1] >= 1.5


def test_external_payload_needs_explicit_comparison_settings(tmp_path):
    run = tmp_path / "run"
    config, neat_config, population, _ = make_run(run)
    genome = next(iter(population.population.values()))
    source = tmp_path / "candidate.pkl"
    source.write_bytes(pickle.dumps({"genome": genome, "config": neat_config,
                                     "schema": VERSION, "generation": 12}))
    with pytest.raises(ValueError, match="External model payloads require"):
        run_holdout(run, [], [("external", source)], seed=32, maps=1, seconds=.1, device="cpu")

    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"config": dataclasses.asdict(config),
                                      "protocol": protocol_settings()}), encoding="utf-8")
    result = run_holdout(run, [], [("external", source)], seed=32, maps=1, seconds=.1,
                         device="cpu", comparison_path=comparison)
    external = next(model for model in result["models"] if model["name"] == "external")
    assert external["source"]["kind"] == "trusted_model_payload"
    assert external["source"]["generation"] == 12
    assert len(external["source"]["payload_sha256"]) == 64


@pytest.mark.parametrize("mutation", ["activation", "aggregation"])
def test_preflight_rejects_phenotype_functions_not_supported_by_batched_network(mutation):
    random.seed(68)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    gene = genome.nodes[neat_config.genome_config.output_keys[0]]
    if mutation == "activation":
        gene.activation = "tanh"
    else:
        gene.aggregation = "max"
    with pytest.raises(ValueError, match="BatchedNetwork requires sigmoid activation and sum aggregation"):
        _validate_genome_config(genome, neat_config)


@pytest.mark.parametrize("mutation", ["input", "output"])
def test_preflight_rejects_unsupported_neat_key_ordering(mutation):
    random.seed(94)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    if mutation == "input":
        neat_config.genome_config.input_keys[0] = 42
    else:
        neat_config.genome_config.output_keys.reverse()
    with pytest.raises(ValueError, match="input/output keys do not match BatchedNetwork ordering"):
        _validate_genome_config(genome, neat_config)


def test_runtime_provenance_includes_code_and_dependency_versions():
    runtime = _runtime_provenance()
    assert runtime["python_version"] and runtime["python_build"]
    assert runtime["torch_version"] and runtime["neat_python_version"]
    assert all(len(value) == 64 for value in runtime["code_sha256"].values())
