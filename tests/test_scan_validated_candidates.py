import copy
import json
import random
import pickle
from pathlib import Path
import hashlib

import neat
import numpy as np
import pytest

from scripts import scan_validated_candidates as scanner
from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.schema import contract


def test_select_top_five_uses_recorded_scores_and_stable_tie_order():
    episode = {
        "generation": 4,
        "genome_ids": [10, 20, 30, 40, 50, 60],
        "score": [8.0, 10.0, 10.0, -2.0, 3.0, 4.0],
    }

    selected = scanner.select_top_five(episode, 4)

    assert [row["genome_id"] for row in selected] == [20, 30, 10, 60, 50]
    assert [row["rank"] for row in selected] == [1, 2, 3, 4, 5]
    assert [row["historical_score_for_selection_only"] for row in selected] == [10.0, 10.0, 8.0, 4.0, 3.0]


def test_deduplication_uses_genotype_hash_and_retains_witness_alias():
    random.seed(201)
    neat_config = load_config(3)
    population = neat.Population(neat_config)
    first, second = list(population.population.values())[:2]
    genotype_copy = copy.deepcopy(first)
    genotype_copy.key = first.key + 10000
    candidates = [
        {"name": "G0-rank1", "genome": first, "source": {"generation": 0},
         "historical_score_for_selection_only": 12.0},
        {"name": "G45-frozen-witness", "genome": genotype_copy,
         "source": {"generation": 44, "display_generation": 45, "genome_id": 8270, "kind": "witness"}},
        {"name": "G4-rank2", "genome": second, "source": {"generation": 4},
         "historical_score_for_selection_only": 11.0},
    ]

    unique = scanner.deduplicate_candidates(candidates)

    assert [row["name"] for row in unique] == ["G0-rank1", "G4-rank2"]
    assert len(unique[0]["genome_sha256"]) == 64
    assert unique[0]["aliases"] == [{
        "name": "G45-frozen-witness",
        "source": {"generation": 44, "display_generation": 45, "genome_id": 8270, "kind": "witness"},
        "historical_score_for_selection_only": None,
    }]


def test_frozen_witness_requires_checkpoint_index_and_genome_identity(monkeypatch):
    random.seed(902)
    neat_config = load_config(3)
    genome = next(iter(neat.Population(neat_config).population.values()))
    genome.key = 8270
    model = {"name": "G45-frozen-witness", "genome": genome, "neat_config": neat_config,
             "source": {"generation": 44, "genome_id": 8270, "sensor_version": "legacy-v1"}}
    monkeypatch.setattr(scanner, "_payload_model", lambda label, path: copy.deepcopy(model))

    witness = scanner._load_witness(Path("frozen.pkl"), "legacy-v1")
    assert witness["name"] == "G45-frozen-witness"
    assert witness["source"]["generation"] == 44
    assert witness["source"]["display_generation"] == 45

    wrong_id = copy.deepcopy(model)
    wrong_id["source"]["genome_id"] = 999
    monkeypatch.setattr(scanner, "_payload_model", lambda label, path: wrong_id)
    with pytest.raises(ValueError, match="genome 8270"):
        scanner._load_witness(Path("wrong.pkl"), "legacy-v1")


def test_cpu_scan_serializes_real_fixed_validation_shape_and_freezes_best(monkeypatch):
    random.seed(482)
    neat_config = load_config(3)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    cloned_witness = copy.deepcopy(genome)
    cloned_witness.key = 8270
    config = SimConfig(maps=32, worms=4, foods=12, body_points=8, preys=0,
                       arena_radius=400., arena_variation=0.)
    comparison = {
        "config_sha256": "a" * 64,
        "protocol": {"version": "test-protocol"},
        "schema": contract("legacy-v1"),
        "schema_version": contract("legacy-v1")["version"],
        "reward_version": config.reward_version,
    }

    def generation_models(run, generation, sensor_version):
        return [{"name": f"G{generation}-rank1-ID{genome.key}",
                 "genome": genome, "neat_config": neat_config,
                 "source": {"kind": "generation_episode_top5", "generation": generation,
                            "rank_within_generation": 1, "genome_id": genome.key,
                            "sensor_version": sensor_version},
                 "historical_score_for_selection_only": 100. - generation}]

    witness = {"name": "G45-frozen-witness", "genome": cloned_witness,
               "neat_config": neat_config,
               "source": {"kind": "frozen_g45_payload_witness", "generation": 44,
                          "display_generation": 45, "genome_id": 8270,
                          "sensor_version": "legacy-v1"}}
    monkeypatch.setattr(scanner, "_load_simulation_config", lambda *args: (config, comparison))
    monkeypatch.setattr(scanner, "_read_json", lambda path: contract("legacy-v1"))
    monkeypatch.setattr(scanner, "_load_generation_candidates", generation_models)
    monkeypatch.setattr(scanner, "_load_witness", lambda *args: copy.deepcopy(witness))
    validation_calls = []
    written = {}

    def fake_fixed_validation(genome_arg, config_arg, neat_config_arg, device, *, seconds, maps):
        validation_calls.append((genome_arg.key, device, seconds, maps))
        return {
            "fitness": 8.5,
            "alive": 0.75,
            "food_gain": 4.25,
            "kills": 0.5,
            "metrics": {"age": 12.75, "turn_degrees": 22.0},
            "standard_error": 0.125,
            "seconds": seconds,
            "maps": maps,
            "seed": scanner.VALIDATION_SEED,
            "per_map": {"fitness": np.arange(maps, dtype=np.float32),
                        "alive": np.ones(maps, dtype=np.bool_)},
        }

    monkeypatch.setattr(scanner, "fixed_validation", fake_fixed_validation)

    def capture_write(path, payload):
        path = Path(path).resolve()
        assert path not in written
        written[path] = payload

    original_file_hash = scanner._file_hash

    def captured_or_real_hash(path):
        path = Path(path).resolve()
        return hashlib.sha256(written[path]).hexdigest() if path in written else original_file_hash(path)

    output = Path("virtual-scan-output").resolve()
    monkeypatch.setattr(scanner, "_prepare_output_dir", lambda path: output)
    monkeypatch.setattr(scanner, "_write_new", capture_write)
    monkeypatch.setattr(scanner, "_file_hash", captured_or_real_hash)
    result = scanner.run_scan(Path("virtual-run"), Path("frozen.pkl"), output, "cpu")

    assert len(validation_calls) == 1  # All three generation rows and G45 collapse to one genotype.
    assert validation_calls[0] == (genome.key, "cpu", 90.0, 32)
    report = json.loads(written[output / "scan.json"].decode("utf-8"))
    assert report["models"][0]["validation"]["per_map"]["fitness"] == list(map(float, range(32)))
    assert report["models"][0]["aliases"][-1]["name"] == "G45-frozen-witness"
    markdown = written[output / "scan.md"].decode("utf-8")
    assert "12.750" in markdown
    assert "G45-frozen-witness" in markdown
    assert result["best_candidate"]["payload_sha256"]
    payload = pickle.loads(written[output / "best-candidate.pkl"])
    assert payload["genome"].key == genome.key
    assert payload["selection"]["aliases"][-1]["name"] == "G45-frozen-witness"
    assert output / "comparison-config.json" in written
