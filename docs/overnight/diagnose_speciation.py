"""Read-only CPU reproduction diagnostics for a saved SlitherAI NEAT run.

Uses saved per-genome episode outcomes as the Population.run fitness callback,
so it performs no simulation or GPU evaluation. Reporter writes and generated
checkpoints go only to a temporary directory. The only live-run inputs are
checkpoint-N, episodes/generation-N.json, settings.json, and history.jsonl.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import itertools
import json
import math
import os
import pickle
import random
import statistics
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import neat
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slitherai import train as train_module  # noqa: E402
from slitherai.config import SimConfig  # noqa: E402
from slitherai.species_metrics import SpeciesReporter  # noqa: E402


def read_checkpoint(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def saved_scores(run: Path, generation: int):
    with (run / "episodes" / f"generation-{generation:04d}.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def assign_saved_fitness(population, episode, history_row):
    """Match Trainer.evaluate's saved score/anchor/behavior assignments."""
    ids = episode["genome_ids"]
    index = {int(gid): i for i, gid in enumerate(ids)}
    metrics = {name: np.asarray(values, dtype=np.float64) for name, values in episode["metrics"].items()}
    if set(index) != set(population):
        raise ValueError("Saved episode genome ids do not match checkpoint population")
    for gid, genome in population.items():
        i = index[gid]
        genome.fitness = float(episode["score"][i])
        genome.anchor_fitness = float(episode["anchor_score"][i])
        genome.behavior = {name: float(values[i].mean()) for name, values in metrics.items()}
    return history_row["evaluation"]


def gene_dict(gene):
    return {key: value for key, value in vars(gene).items() if key != "key"}


def genome_state(genome):
    return {
        "fitness": genome.fitness,
        "anchor_fitness": getattr(genome, "anchor_fitness", None),
        "behavior": getattr(genome, "behavior", None),
        "nodes": {int(k): gene_dict(v) for k, v in sorted(genome.nodes.items())},
        "connections": {str(tuple(k)): gene_dict(v) for k, v in sorted(genome.connections.items())},
    }


def species_state(species_set):
    result = {}
    for sid, species in sorted(species_set.species.items()):
        result[int(sid)] = {
            "created": species.created,
            "last_improved": species.last_improved,
            "representative": species.representative.key,
            "members": sorted(int(k) for k in species.members),
            "fitness": species.fitness,
            "adjusted_fitness": species.adjusted_fitness,
            "fitness_history": list(species.fitness_history),
            "recent_score": getattr(species, "recent_score", None),
            "progress_delta": getattr(species, "progress_delta", None),
            "stagnation_eligible": getattr(species, "stagnation_eligible", None),
            "protected": getattr(species, "protected", None),
        }
    return result


def counter_next(counter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reduced = counter.__reduce__()
    return list(reduced[1])


def node_index_next(genome_config):
    return None if genome_config.node_indexer is None else counter_next(genome_config.node_indexer)


def tracker_state(tracker):
    if tracker is None:
        return None
    state = vars(tracker)
    innovations = state.get("generation_innovations", {})
    return {
        "global_counter": state.get("global_counter"),
        "generation_innovations": sorted((repr(key), value) for key, value in innovations.items()),
    }


def component_distance(a, b, genome_config):
    """Return the two components used by neat.DefaultGenome.distance."""
    disjoint_coefficient = genome_config.compatibility_disjoint_coefficient
    node_sum = 0.0
    if a.nodes or b.nodes:
        disjoint = sum(k not in a.nodes for k in b.nodes)
        for k, gene in a.nodes.items():
            other = b.nodes.get(k)
            if other is None:
                disjoint += 1
            else:
                node_sum += gene.distance(other, genome_config)
        node_sum = (node_sum + disjoint_coefficient * disjoint) / max(len(a.nodes), len(b.nodes))
    conn_sum = 0.0
    if a.connections or b.connections:
        disjoint = sum(k not in a.connections for k in b.connections)
        for k, gene in a.connections.items():
            other = b.connections.get(k)
            if other is None:
                disjoint += 1
            else:
                conn_sum += gene.distance(other, genome_config)
        conn_sum = (conn_sum + disjoint_coefficient * disjoint) / max(len(a.connections), len(b.connections))
    return node_sum, conn_sum


def mismatch_summary(actual, expected):
    actual_ids, expected_ids = set(actual), set(expected)
    result = {"population_ids_match": actual_ids == expected_ids,
              "actual_population": len(actual), "expected_population": len(expected),
              "matching_genomes": 0, "mismatching_genome_ids": [],
              "first_divergent_genome": None}
    for gid in sorted(actual_ids & expected_ids):
        left, right = genome_state(actual[gid]), genome_state(expected[gid])
        if left == right:
            result["matching_genomes"] += 1
            continue
        diff = {
            "id": gid,
            "nodes_actual": sorted(left["nodes"]),
            "nodes_expected": sorted(right["nodes"]),
            "node_keys_only_actual": sorted(set(left["nodes"]) - set(right["nodes"])),
            "node_keys_only_expected": sorted(set(right["nodes"]) - set(left["nodes"])),
            "connection_keys_only_actual": sorted(set(left["connections"]) - set(right["connections"])),
            "connection_keys_only_expected": sorted(set(right["connections"]) - set(left["connections"])),
            "fitness_equal": left["fitness"] == right["fitness"],
            "anchor_fitness_equal": left["anchor_fitness"] == right["anchor_fitness"],
        }
        if result["first_divergent_genome"] is None:
            result["first_divergent_genome"] = diff
        result["mismatching_genome_ids"].append(gid)
    return result


def graph_equal_up_to_hidden_renaming(actual, expected, config):
    """Compare expressed genes ignoring hidden-node labels and innovation IDs."""
    output_keys = set(config.genome_config.output_keys)
    actual_hidden = sorted(set(actual.nodes) - output_keys)
    expected_hidden = sorted(set(expected.nodes) - output_keys)
    if len(actual_hidden) != len(expected_hidden):
        return False
    for mapped in itertools.permutations(expected_hidden):
        mapping = dict(zip(actual_hidden, mapped))
        if any(gene_dict(actual.nodes[k]) != gene_dict(expected.nodes[mapping.get(k, k)])
               for k in actual.nodes):
            continue
        actual_edges = {}
        for key, gene in actual.connections.items():
            source, target = key
            actual_edges[(mapping.get(source, source), mapping.get(target, target))] = (gene.weight, gene.enabled)
        expected_edges = {key: (gene.weight, gene.enabled) for key, gene in expected.connections.items()}
        if actual_edges == expected_edges:
            return True
    return False


def activation_difference(actual, expected, actual_config, expected_config):
    """Compare recurrent CPU activations on deterministic inputs for five ticks."""
    left = neat.nn.RecurrentNetwork.create(actual, actual_config)
    right = neat.nn.RecurrentNetwork.create(expected, expected_config)
    maximum = 0.0
    exactly_equal = True
    input_count = len(actual_config.genome_config.input_keys)
    for tick in range(5):
        inputs = [math.sin((i + 1) * (tick + 1) * 0.017) for i in range(input_count)]
        a, b = left.activate(inputs), right.activate(inputs)
        deltas = [abs(x-y) for x, y in zip(a, b)]
        maximum = max(maximum, *deltas)
        exactly_equal = exactly_equal and a == b
    return {"max_abs_output_difference": maximum, "exact_outputs_equal": exactly_equal}


def replay_speciation(source_species, next_population, genome_config, generation, threshold,
                      minimum_species_size, pop_size):
    """Mirror AdaptiveSpeciesSet's representative selection and assignment order."""
    remaining = sorted(next_population)
    representatives = {}
    for sid, species in sorted(source_species.species.items()):
        key = min(remaining, key=lambda k: (species.representative.distance(next_population[k], genome_config), k))
        representatives[sid] = next_population[key]
        remaining.remove(key)
    capacity = max(1, pop_size // max(minimum_species_size, 1))
    next_sid = counter_next(source_species.indexer)[0]
    created = []
    for key in remaining:
        genome = next_population[key]
        choices = sorted((genome.distance(rep, genome_config), sid) for sid, rep in representatives.items())
        if choices and (choices[0][0] < threshold or len(representatives) >= capacity):
            continue
        nearest_total, nearest_sid = choices[0] if choices else (None, None)
        node, connection = component_distance(representatives[nearest_sid], genome, genome_config) if nearest_sid is not None else (None, None)
        created.append({"species_id": next_sid, "founder_genome_id": key,
                        "nearest_existing_species_id": nearest_sid,
                        "node_distance": node, "connection_distance": connection,
                        "total_distance": nearest_total})
        representatives[next_sid] = genome
        next_sid += 1
    return representatives, created


def _validation_stub(*args, **kwargs):
    """Avoid rerunning the saved validation simulations during a CPU audit."""
    if args and hasattr(args[0], "fitness"):
        pass
    return dict(fitness=0.0, alive=0.0, food_gain=0.0, kills=0.0,
                metrics={}, standard_error=0.0, seconds=90.0, maps=32,
                seed=938271, per_map={})


def replay_transition(run: Path, generation: int, settings, history_rows):
    source_path = run / f"checkpoint-{generation}"
    target_path = run / f"checkpoint-{generation + 1}"
    saved_run_generation, _, _, _, _ = read_checkpoint(source_path)
    target_generation, target_config, target_population, target_species, target_random = read_checkpoint(target_path)
    if saved_run_generation != generation or target_generation != generation + 1:
        raise ValueError(f"Unexpected checkpoint labels for transition {generation}")

    tmp_context = tempfile.TemporaryDirectory(prefix="slitherai-neat-replay-")
    try:
        trainer = train_module.Trainer(
            SimConfig(**settings["config"]), Path(tmp_context.name), device="cpu",
            seed=int(settings["seed"]), validation_every=int(settings.get("validation_every", 5)))
        trainer.population_size = int(settings["population"])
        trainer.generations = 1
        trainer.episode_seconds = float(settings["seconds"])
        # Mirror Trainer.train's process initialization followed by checkpoint restore.
        random.seed(trainer.seed)
        np.random.seed(trainer.seed)
        population = neat.Checkpointer.restore_checkpoint(str(source_path))
        trainer.base_generation = population.generation
        trainer.species_tracker = SpeciesReporter(Path(tmp_context.name), population)
        population.add_reporter(trainer.species_tracker)
        population.add_reporter(train_module.TrainingReporter(trainer))
        serialization_events = []
        original_save_genome = trainer.save_genome

        def traced_save_genome(genome, config, name):
            before = node_index_next(config.genome_config)
            result = original_save_genome(genome, config, name)
            after = node_index_next(config.genome_config)
            serialization_events.append({"source": f"TrainingReporter.save_genome:{name}",
                                         "node_indexer_next_before": before,
                                         "node_indexer_next_after": after})
            return result

        trainer.save_genome = traced_save_genome
        checkpoint_reporter = train_module.AtomicCheckpointer(
            1, filename_prefix=str(Path(tmp_context.name) / "checkpoint-"))
        original_checkpoint_save = checkpoint_reporter.save_checkpoint

        def traced_checkpoint_save(config, genomes, species_set, generation_number):
            before = node_index_next(config.genome_config)
            result = original_checkpoint_save(config, genomes, species_set, generation_number)
            after = node_index_next(config.genome_config)
            serialization_events.append({"source": "AtomicCheckpointer.save_checkpoint",
                                         "node_indexer_next_before": before,
                                         "node_indexer_next_after": after})
            return result

        checkpoint_reporter.save_checkpoint = traced_checkpoint_save
        population.add_reporter(checkpoint_reporter)

        old_species_ids = set(population.species.species)
        episode = saved_scores(run, generation)
        history_row = history_rows[generation]
        validation_calls = []
        original_validation = train_module.fixed_validation

        def checked_stub(*args, **kwargs):
            before = random.getstate()
            result = _validation_stub(*args, **kwargs)
            validation_calls.append({"policy": kwargs.get("policy", "neat"),
                                     "python_rng_unchanged": before == random.getstate()})
            return result

        train_module.fixed_validation = checked_stub
        try:
            def load_scores(genomes, _config):
                trainer.evaluation_metrics = assign_saved_fitness(population.population, episode, history_row)
                trainer.progress = {"completed_episodes": len(episode["genome_ids"])*len(episode["scenarios"]),
                                    "total_episodes": len(episode["genome_ids"])*len(episode["scenarios"])}
            # Capture normal reporter console output to keep the JSON audit compact.
            with contextlib.redirect_stdout(io.StringIO()):
                population.run(load_scores, 1)
        finally:
            train_module.fixed_validation = original_validation

        generated_checkpoint = Path(tmp_context.name) / f"checkpoint-{generation + 1}"
        if not generated_checkpoint.exists():
            raise RuntimeError("Production Checkpointer reporter did not emit the expected next checkpoint")
        generated_generation, generated_config, generated_population, generated_species, generated_random = read_checkpoint(generated_checkpoint)
        population_result = mismatch_summary(generated_population, target_population)
        generated_species_state = species_state(generated_species)
        target_species_state = species_state(target_species)
        species_diffs = [sid for sid in sorted(set(generated_species_state) | set(target_species_state))
                         if generated_species_state.get(sid) != target_species_state.get(sid)]

        genome_indexers = {
            "generated": counter_next(population.reproduction.genome_indexer),
            "expected": counter_next(neat.Checkpointer.restore_checkpoint(str(target_path)).reproduction.genome_indexer),
        }
        generated_tracker = tracker_state(population.reproduction.innovation_tracker)
        expected_tracker = tracker_state(target_config.genome_config.innovation_tracker)
        generated_node_next = node_index_next(generated_config.genome_config)
        expected_node_next = node_index_next(target_config.genome_config)
        generated_innovations = dict(generated_tracker["generation_innovations"])
        expected_innovations = dict(expected_tracker["generation_innovations"])
        population_result.update({
            "generated_checkpoint_generation": generated_generation,
            "expected_checkpoint_generation": target_generation,
            "species_mismatch_ids": species_diffs,
            "compatibility_threshold_generated": generated_config.species_set_config.compatibility_threshold,
            "compatibility_threshold_expected": target_config.species_set_config.compatibility_threshold,
            "innovation_tracker_equal": generated_tracker == expected_tracker,
            "innovation_tracker_generated_counter": generated_tracker["global_counter"],
            "innovation_tracker_expected_counter": expected_tracker["global_counter"],
            "innovation_tracker_generation_key_mismatches": len(set(generated_innovations) ^ set(expected_innovations)),
            "python_rng_equal": generated_random == target_random,
            "node_indexer_next": {"source": node_index_next(read_checkpoint(source_path)[1].genome_config),
                                  "generated": generated_node_next, "expected": expected_node_next},
            "species_indexer_generated_next": counter_next(generated_species.indexer),
            "species_indexer_expected_next": counter_next(target_species.indexer),
            "genome_indexer_next": genome_indexers,
            "validation_calls": validation_calls,
            "validation_simulation": "stubbed; caller reads saved episode fitness only",
            "config_pickle_node_indexer_events": serialization_events,
            "species_equal": not species_diffs,
            "new_species_ids": sorted(set(target_species.species) - old_species_ids),
        })
        if population_result["mismatching_genome_ids"]:
            graph_equal = []
            activation_diffs = []
            for gid in population_result["mismatching_genome_ids"]:
                graph_equal.append(graph_equal_up_to_hidden_renaming(
                    generated_population[gid], target_population[gid], generated_config))
                activation_diffs.append(activation_difference(
                    generated_population[gid], target_population[gid], generated_config, target_config))
            population_result["mismatches_graph_equal_up_to_hidden_renaming"] = sum(graph_equal)
            population_result["mismatches_cpu_activation_equal"] = sum(x["exact_outputs_equal"] for x in activation_diffs)
            population_result["mismatches_cpu_activation_max_abs_delta"] = max(
                x["max_abs_output_difference"] for x in activation_diffs)

        if generation == 0:
            pool_sizes = []
            source_data = read_checkpoint(source_path)
            source_population = source_data[2]
            source_species = source_data[3]
            assign_saved_fitness(source_population, episode, history_row)
            for sid, species in sorted(source_species.species.items()):
                ordered = sorted(species.members.items(), reverse=True,
                                 key=lambda pair: (pair[1].fitness, pair[0]))
                cutoff = max(math.ceil(population.config.reproduction_config.survival_threshold * len(ordered)), 2)
                pool = [gid for gid, _ in ordered[:cutoff]]
                pool_sizes.append(len(pool))
            actual_parent_pairs = list(population.reproduction.ancestors.values())
            selected_parent_ids = {gid for pair in actual_parent_pairs for gid in pair}
            child_node_adds = 0
            child_connection_adds = 0
            for child_id, parents in population.reproduction.ancestors.items():
                parent_nodes = set(source_population[parents[0]].nodes) | set(source_population[parents[1]].nodes)
                introduced = set(generated_population[child_id].nodes) - parent_nodes
                if introduced:
                    child_node_adds += 1
                parent_connections = set(source_population[parents[0]].connections) | set(source_population[parents[1]].connections)
                if set(generated_population[child_id].connections) - parent_connections:
                    child_connection_adds += 1
            population_result["generation0_parent_pool_sizes"] = pool_sizes
            population_result["generation0_parent_pool_total"] = sum(pool_sizes)
            population_result["generation0_parent_pool_unique_actual_parents"] = len(selected_parent_ids)
            population_result["generation0_child_parent_pairs"] = len(actual_parent_pairs)
            population_result["generation0_same_parent_child_pairs"] = sum(a == b for a, b in actual_parent_pairs)
            population_result["generation0_offspring_with_new_node_vs_both_parents"] = child_node_adds
            population_result["generation0_offspring_with_new_connection_vs_both_parents"] = child_connection_adds

        if generation == 0:
            retained_reps, algorithm_founders = replay_speciation(
                source_data[3], generated_population, target_config.genome_config,
                generation + 1, target_config.species_set_config.compatibility_threshold,
                max(target_config.reproduction_config.min_species_size,
                    target_config.reproduction_config.elitism, 1), target_config.pop_size)
            expected_founders = []
            for row in algorithm_founders:
                founder = generated_population[row["founder_genome_id"]]
                child_parents = population.reproduction.ancestors.get(founder.key)
                parent_node_keys, parent_connection_keys = set(), set()
                if child_parents:
                    for parent in child_parents:
                        parent_node_keys.update(source_population[parent].nodes)
                        parent_connection_keys.update(source_population[parent].connections)
                row["new_node_keys_vs_parents"] = sorted(set(founder.nodes) - parent_node_keys)
                row["new_connection_keys_vs_parents"] = sorted(map(str, set(founder.connections) - parent_connection_keys))
                new_nodes = row.pop("new_node_keys_vs_parents")
                new_connections = row.pop("new_connection_keys_vs_parents")
                row["new_node_count_vs_parents"] = len(new_nodes)
                row["new_connection_count_vs_parents"] = len(new_connections)
                row["founder_mutation_shape"] = (
                    "node+connection" if new_nodes and new_connections else
                    "node" if new_nodes else "connection" if new_connections else "none")
                expected_founders.append(row)
            new_ids = set(target_species.species) - old_species_ids
            target_founders = {target_species.species[sid].representative.key: sid for sid in new_ids}
            for row in expected_founders:
                row["checkpoint_species_id"] = target_founders.get(row["founder_genome_id"])
            retained_old_reps = [(sid, retained_reps[sid]) for sid in sorted(old_species_ids)
                                 if sid in retained_reps]
            nearest_components = []
            for genome in generated_population.values():
                nearest = min((sum(component_distance(rep, genome, target_config.genome_config)), sid,
                               component_distance(rep, genome, target_config.genome_config))
                              for sid, rep in retained_old_reps)
                nearest_components.append((nearest[2], nearest[1]))
            total_distances = [sum(value[0]) for value in nearest_components]
            population_result["generation0_founder_distances"] = expected_founders
            population_result["generation0_speciation_replay_matches_checkpoint"] = {
                "founder_genome_ids": sorted(row["founder_genome_id"] for row in expected_founders)
                    == sorted(target_founders),
                "species_ids": all(row["checkpoint_species_id"] == row["species_id"] for row in expected_founders),
            }
            founder_totals = [row["total_distance"] for row in expected_founders]
            population_result["generation0_founder_distance_range"] = [min(founder_totals), max(founder_totals)] if founder_totals else None
            population_result["generation0_population_nearest_old_rep_median"] = {
                "node": statistics.median(pair[0][0] for pair in nearest_components),
                "connection": statistics.median(pair[0][1] for pair in nearest_components),
                "total": statistics.median(total_distances),
                "at_or_above_threshold": sum(d >= target_config.species_set_config.compatibility_threshold
                                             for d in total_distances),
            }
        return population_result
    finally:
        tmp_context.cleanup()


def allocation_summary(run: Path, generation: int = 3):
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    episode = saved_scores(run, generation)
    _, _, source_population, _, _ = read_checkpoint(run / f"checkpoint-{generation}")
    history_rows = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_metrics = next(row["evaluation"] for row in history_rows if row["generation"] == generation)
    outputs = []
    for minimum in (4, 2):
        random.seed(int(settings["seed"]))
        pop = neat.Checkpointer.restore_checkpoint(str(run / f"checkpoint-{generation}"))
        assign_saved_fitness(pop.population, episode, {"evaluation": eval_metrics})
        pop.config.reproduction_config.min_species_size = minimum
        source_sizes = {sid: len(species.members) for sid, species in pop.species.species.items()}
        captured = {"species_ids": None, "pre_exact": None, "spawn": None}
        original_stagnation_update = pop.reproduction.stagnation.update

        def capture_stagnation(species_set, gen):
            result = original_stagnation_update(species_set, gen)
            captured["species_ids"] = [sid for sid, _, stagnant in result if not stagnant]
            return result

        pop.reproduction.stagnation.update = capture_stagnation
        original_compute_spawn = pop.reproduction.compute_spawn

        def capture_compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, floor):
            result = original_compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, floor)
            captured["pre_exact"] = list(result)
            return result

        pop.reproduction.compute_spawn = capture_compute_spawn
        original_adjust_spawn = pop.reproduction._adjust_spawn_exact

        def capture_adjust_spawn(spawn_amounts, pop_size, floor):
            result = original_adjust_spawn(spawn_amounts, pop_size, floor)
            captured["spawn"] = list(result)
            return result

        pop.reproduction._adjust_spawn_exact = capture_adjust_spawn
        effective_minimum = max(minimum, pop.config.reproduction_config.elitism)
        elites = pop.config.reproduction_config.elitism
        new_population = pop.reproduction.reproduce(pop.config, pop.species, pop.config.pop_size, generation)
        active_ids = captured["species_ids"]
        spawn = captured["spawn"]
        pre_exact = captured["pre_exact"]
        source_size_by_active_id = [source_sizes[sid] for sid in active_ids]
        actual_elite_copies = sum(min(elites, size) for size in source_size_by_active_id)
        children = [max(0, amount-min(elites, size))
                    for amount, size in zip(spawn, source_size_by_active_id)]
        actual_child_count = len(pop.reproduction.ancestors)
        actual_elite_ids = set(new_population) - set(pop.reproduction.ancestors)
        if actual_elite_copies != len(actual_elite_ids) or actual_child_count != sum(children):
            raise AssertionError("Predicted per-species reproduction counts disagree with actual offspring")
        total = sum(spawn)
        shares = [amount/total for amount in spawn]
        effective_species = math.exp(-sum(p*math.log(p) for p in shares if p > 0))
        child_shares = [n/actual_child_count for n in children if n > 0]
        effective_child_species = math.exp(-sum(p*math.log(p) for p in child_shares)) if actual_child_count else 0.0
        distribution = dict(sorted(Counter(spawn).items()))
        outputs.append({
            "min_species_size": minimum,
            "effective_minimum": effective_minimum,
            "species": len(active_ids),
            "spawn_sum": total,
            "spawn_distribution": {str(k): v for k, v in distribution.items()},
            "species_at_minimum": sum(amount == effective_minimum for amount in spawn),
            "elite_only_species": sum(child_count == 0 for child_count in children),
            "elite_copies": actual_elite_copies,
            "actual_elite_ids": len(actual_elite_ids),
            "mutant_children": actual_child_count,
            "top4_spawn": sum(sorted(spawn, reverse=True)[:4]),
            "top8_spawn": sum(sorted(spawn, reverse=True)[:8]),
            "top4_spawn_share": sum(sorted(spawn, reverse=True)[:4]) / total,
            "top8_spawn_share": sum(sorted(spawn, reverse=True)[:8]) / total,
            "effective_species_total_spawn": effective_species,
            "effective_species_mutant_children": effective_child_species,
            "largest_spawn_share": max(spawn) / total,
            "pre_exact_spawn_sum": sum(pre_exact),
            "per_species": [
                {"species_id": sid, "members": source_size_by_active_id[i],
                 "spawn": spawn[i], "elite_copies": min(elites, source_size_by_active_id[i]),
                 "mutant_children": children[i]}
                for i, sid in enumerate(active_ids)
            ],
        })
    outputs[0]["min2_vs_min4_spawn_identical"] = outputs[0]["per_species"] == outputs[1]["per_species"]
    by_id = [{row["species_id"]: row["spawn"] for row in output["per_species"]} for output in outputs]
    outputs[0]["changed_species_allocations"] = [
        {"species_id": sid, "min4_spawn": by_id[0][sid], "min2_spawn": by_id[1][sid],
         "delta_min2_minus_min4": by_id[1][sid]-by_id[0][sid]}
        for sid in sorted(by_id[0]) if by_id[0][sid] != by_id[1][sid]
    ]
    for output in outputs:
        output.pop("per_species")
    return {"generation": generation, "source_checkpoint": str(run / f"checkpoint-{generation}"),
            "comparison": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "20260924-163112-852097")
    parser.add_argument("--transition", type=int, action="append", help="replay N -> N+1; repeat to select; default 0,1,2")
    parser.add_argument("--allocation-generation", type=int, default=3)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--no-allocation", action="store_true")
    parser.add_argument("--output", type=Path, help="also save the compact JSON result outside the run")
    args = parser.parse_args()
    run = args.run.resolve()
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    history_rows = {row["generation"]: row for row in
                    (json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines())}
    result = {"run": str(run), "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
              "neat_python_version": getattr(neat, "__version__", "unknown"),
              "replay": [], "allocation": None}
    transitions = args.transition if args.transition is not None else [0, 1, 2]
    if not args.no_replay:
        for generation in transitions:
            result["replay"].append(replay_transition(run, generation, settings, history_rows))
    if not args.no_allocation:
        result["allocation"] = allocation_summary(run, args.allocation_generation)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
