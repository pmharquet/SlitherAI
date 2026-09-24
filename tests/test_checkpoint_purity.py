import contextlib
import gzip
import io
import pickle
import random
import shutil
import warnings

import neat
import numpy as np
import pytest

from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.species_metrics import SpeciesReporter
from slitherai.train import AtomicCheckpointer, Trainer, TrainingReporter


def counter_next(counter):
    if counter is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return counter.__reduce__()[1][0]


def population_with_node_counter(seed=731, size=16, mutate_initial=True):
    random.seed(seed)
    np.random.seed(seed)
    config = load_config(size)
    config.genome_config.node_add_prob = 1.0
    config.genome_config.node_delete_prob = 0.0
    config.genome_config.conn_add_prob = 0.0
    config.genome_config.conn_delete_prob = 0.0
    population = neat.Population(config)
    genome = next(iter(population.population.values()))
    if mutate_initial:
        genome.mutate(config.genome_config)
        assert len(genome.nodes) > config.genome_config.num_outputs
    assert len(genome.connections) >= 20
    if mutate_initial:
        assert counter_next(config.genome_config.node_indexer) is not None
    return population


def test_champion_and_checkpoint_serialization_preserve_live_node_counter(tmp_path):
    population = population_with_node_counter()
    config = population.config
    expected = counter_next(config.genome_config.node_indexer)
    trainer = Trainer(SimConfig(), tmp_path, device="cpu", validation_every=0)
    trainer.generation = 7
    trainer.save_genome(next(iter(population.population.values())), config, "champion")
    assert counter_next(config.genome_config.node_indexer) == expected

    champion = pickle.loads((tmp_path / "champion.pkl").read_bytes())
    assert counter_next(champion["config"].genome_config.node_indexer) == expected
    assert champion["schema"]

    reporter = AtomicCheckpointer(1, filename_prefix=str(tmp_path / "checkpoint-"))
    reporter.save_checkpoint(config, population.population, population.species, 7)
    assert counter_next(config.genome_config.node_indexer) == expected
    with gzip.open(tmp_path / "checkpoint-7", "rb") as stream:
        generation, saved_config, genomes, species, rng_state = pickle.load(stream)
    assert generation == 7
    assert counter_next(saved_config.genome_config.node_indexer) == expected
    assert set(genomes) == set(population.population)
    assert species.reporters.reporters == []
    assert rng_state == random.getstate()


@pytest.mark.parametrize("artifact", ["champion", "checkpoint"])
def test_failed_serialization_does_not_consume_live_node_counter(tmp_path, monkeypatch, artifact):
    population = population_with_node_counter(seed=732)
    config = population.config
    expected = counter_next(config.genome_config.node_indexer)
    trainer = Trainer(SimConfig(), tmp_path, device="cpu", validation_every=0)
    reporter = AtomicCheckpointer(1, filename_prefix=str(tmp_path / "checkpoint-"))
    original_dump = pickle.dump
    original_dumps = pickle.dumps

    def fail_after_serialization(obj, *args, **kwargs):
        # Exercise config __getstate__ and write the temporary artifact before
        # simulating a failure that prevents replacing the previous final file.
        if artifact == "checkpoint":
            original_dump(obj, *args, **kwargs)
        else:
            original_dumps(obj, *args, **kwargs)
        raise OSError("injected serialization/write failure")

    if artifact == "champion":
        final_file = tmp_path / "champion.pkl"
        network_file = tmp_path / "champion-network.json"
        final_file.write_bytes(b"prior champion")
        network_file.write_bytes(b"prior network")
        monkeypatch.setattr("slitherai.train.pickle.dumps", fail_after_serialization)
        save = lambda: trainer.save_genome(next(iter(population.population.values())), config, "champion")
    else:
        final_file = tmp_path / "checkpoint-0"
        final_file.write_bytes(b"prior checkpoint")
        network_file = None
        monkeypatch.setattr("slitherai.train.pickle.dump", fail_after_serialization)
        save = lambda: reporter.save_checkpoint(config, population.population, population.species, 0)

    with pytest.raises(OSError, match="injected"):
        save()
    assert counter_next(config.genome_config.node_indexer) == expected
    assert final_file.read_bytes() == (
        b"prior champion" if artifact == "champion" else b"prior checkpoint"
    )
    if network_file is not None:
        assert network_file.read_bytes() == b"prior network"


def attach_training_reporters(population, run_dir, seed):
    trainer = Trainer(SimConfig(), run_dir, device="cpu", seed=seed, validation_every=0)
    population.add_reporter(SpeciesReporter(run_dir, population))
    population.add_reporter(TrainingReporter(trainer))
    population.add_reporter(AtomicCheckpointer(1, filename_prefix=str(run_dir / "checkpoint-")))
    return trainer


def evaluate_saved_scores(genomes, _config):
    # Stable scores stand in for saved evaluation outcomes; evolution randomness
    # and reporter order are the quantities under test, not game simulation.
    for key, genome in genomes:
        genome.fitness = float(1 + key % 13)
        genome.anchor_fitness = genome.fitness
        genome.behavior = {"food_gain": 0.0, "alive": 0.0}


def gene_state(gene):
    return tuple(sorted(vars(gene).items()))


def checkpoint_signature(path):
    with gzip.open(path, "rb") as stream:
        generation, config, population, species, rng_state = pickle.load(stream)
    gc = config.genome_config
    tracker = gc.innovation_tracker
    genomes = {
        key: (
            genome.fitness,
            tuple((node_key, gene_state(gene)) for node_key, gene in sorted(genome.nodes.items())),
            tuple((connection_key, gene_state(gene))
                  for connection_key, gene in sorted(genome.connections.items())),
        )
        for key, genome in sorted(population.items())
    }
    species_state = tuple(
        (sid, item.created, item.last_improved, item.representative.key,
         tuple(sorted(item.members)), item.fitness, item.adjusted_fitness,
         tuple(item.fitness_history))
        for sid, item in sorted(species.species.items())
    )
    tracker_state = (
        tracker.global_counter,
        tuple(sorted(tracker.generation_innovations.items())),
    )
    return {
        "generation": generation,
        "genomes": genomes,
        "species": species_state,
        "node_indexer_next": counter_next(gc.node_indexer),
        "species_indexer_next": counter_next(species.indexer),
        "innovation_tracker": tracker_state,
        "rng_state": rng_state,
        "compatibility_threshold": config.species_set_config.compatibility_threshold,
    }


def checkpoint_node_count(path):
    with gzip.open(path, "rb") as stream:
        _, _, population, _, _ = pickle.load(stream)
    return sum(len(genome.nodes) for genome in population.values())


def test_forced_node_add_resume_matches_uninterrupted_two_generations(tmp_path):
    seed = 733
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted_dir.mkdir()
    resumed_dir.mkdir()

    population = population_with_node_counter(seed=seed, size=64, mutate_initial=False)
    config = population.config
    assert config.genome_config.num_inputs == 530
    assert min(len(genome.connections) for genome in population.population.values()) >= 20
    initial_nodes = sum(len(genome.nodes) for genome in population.population.values())
    initial_checkpoint = AtomicCheckpointer(
        1, filename_prefix=str(uninterrupted_dir / "checkpoint-")
    )
    initial_checkpoint.save_checkpoint(config, population.population, population.species, 0)
    shutil.copyfile(uninterrupted_dir / "checkpoint-0", resumed_dir / "checkpoint-0")

    attach_training_reporters(population, uninterrupted_dir, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        population.run(evaluate_saved_scores, 2)
    generation_one_nodes = checkpoint_node_count(uninterrupted_dir / "checkpoint-1")
    generation_two_nodes = checkpoint_node_count(uninterrupted_dir / "checkpoint-2")
    assert generation_one_nodes > initial_nodes
    assert generation_two_nodes > generation_one_nodes

    random.seed(seed)
    np.random.seed(seed)
    resumed = neat.Checkpointer.restore_checkpoint(str(resumed_dir / "checkpoint-0"))
    attach_training_reporters(resumed, resumed_dir, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        resumed.run(evaluate_saved_scores, 1)
    resumed = neat.Checkpointer.restore_checkpoint(str(resumed_dir / "checkpoint-1"))
    attach_training_reporters(resumed, resumed_dir, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        resumed.run(evaluate_saved_scores, 1)

    uninterrupted_signature = checkpoint_signature(uninterrupted_dir / "checkpoint-2")
    resumed_signature = checkpoint_signature(resumed_dir / "checkpoint-2")
    assert uninterrupted_signature == resumed_signature
    assert uninterrupted_signature["generation"] == 2
    assert uninterrupted_signature["node_indexer_next"] > 0
