import gzip
import json
import pickle
import random
import neat
import pytest
from slitherai.network import load_config
from slitherai.species_metrics import SpeciesReporter, composition
from slitherai.train import AtomicCheckpointer


def population_with_three_stagnant_species():
    random.seed(432)
    population = neat.Population(load_config(12, legacy=True))
    population.generation = 20
    members = list(population.population.items())
    population.species.species = {}
    population.species.genome_to_species = {}
    for index in range(3):
        species = neat.species.Species(index+1, 0)
        group = dict(members[index*4:(index+1)*4])
        species.update(next(iter(group.values())), group)
        species.fitness_history = [1000.]
        population.species.species[species.key] = species
        population.species.genome_to_species.update({key: species.key for key in group})
    return population


def evaluate(genomes, config):
    for key, genome in genomes:
        genome.fitness = float(key)


def fingerprint(population):
    return [(key, [(k, n.bias) for k, n in g.nodes.items()],
             [(k, c.weight, c.enabled) for k, c in g.connections.items()])
            for key, g in population.population.items()]


def test_species_telemetry_matches_actual_stagnation_and_does_not_change_evolution(tmp_path):
    population = population_with_three_stagnant_species()
    reporter = SpeciesReporter(tmp_path, population)
    population.add_reporter(reporter)
    population.run(evaluate, 1)
    data = json.loads((tmp_path / 'species.json').read_text())
    rows = data['latest']['rows']
    assert [r['best'] for r in rows] == [4, 8, 12]
    assert [r['mean'] for r in rows] == [2.5, 6.5, 10.5]
    assert rows[0]['removed'] and rows[0]['adjusted_fitness'] is None
    assert [r['id'] for r in rows if r['protected']] == [2, 3]
    assert all(r['stagnant_for'] == 20 for r in rows)
    assert rows[1]['adjusted_fitness'] == pytest.approx((6.5-5)/(12-5))
    assert data['latest']['removed_ids'] == [1]
    assert sum(r['size'] for r in data['current']['rows']) == 12
    assert data['current']['generation'] == 21
    baseline = population_with_three_stagnant_species()
    baseline.run(evaluate, 1)
    assert fingerprint(population) == fingerprint(baseline)
    assert population.species.genome_to_species == baseline.species.genome_to_species


def test_checkpoints_exclude_live_reporters_and_preserve_species_indexer(tmp_path):
    population = population_with_three_stagnant_species()
    runtime = neat.reporting.BaseReporter()
    runtime.unpicklable = lambda: 'GPU inspector and telemetry must not be serialized'
    population.add_reporter(runtime)
    before = population.species.indexer.__reduce__()[1]
    reporter = AtomicCheckpointer(1, filename_prefix=str(tmp_path / 'checkpoint-'))
    reporter.save_checkpoint(population.config, population.population, population.species, 20)
    assert population.species.indexer.__reduce__()[1] == before
    with gzip.open(tmp_path / 'checkpoint-20', 'rb') as stream:
        gen, config, genomes, species, rng = pickle.load(stream)
    assert species.reporters.reporters == []
    assert species.indexer.__reduce__()[1] == before
    assert set(genomes) == set(population.population)
    assert gen == 20 and rng == random.getstate()


def test_resume_backfills_only_membership_without_inventing_fitness_or_changing_rng(tmp_path):
    population = population_with_three_stagnant_species()
    checkpoint = AtomicCheckpointer(1, filename_prefix=str(tmp_path / 'checkpoint-'))
    checkpoint.save_checkpoint(population.config, population.population, population.species, 19)
    rng = random.getstate()
    reporter = SpeciesReporter(tmp_path, population)
    assert random.getstate() == rng
    assert reporter.history[0]['reconstructed']
    assert reporter.history[0]['generation'] == 19
    assert sum(r['size'] for r in reporter.history[0]['rows']) == 12
    assert all('mean' not in r for r in reporter.history[0]['rows'])
    assert composition(population.species, 20)['effective'] == pytest.approx(3.)
