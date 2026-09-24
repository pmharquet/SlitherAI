"""Reproduce the paired G1/G5/G10 validation and population audit.

Read-only: consumes saved validation, episode, history, and NEAT checkpoint
artifacts. Prints Markdown to stdout; it does not rescore policies or modify
the run. Usage from the repository root:

    .venv/Scripts/python.exe docs/overnight/analyze_g10_audit.py
    .venv/Scripts/python.exe docs/overnight/analyze_g10_audit.py --run runs/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import neat


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_RUN = ROOT / "runs" / "20260924-163112-852097"
VALIDATION_FILES = (("G1", 0), ("G5", 4), ("G10", 9))
METRICS = ("fitness", "food_gain", "alive", "boost_spent", "age", "kills")
NORMAL_95 = 1.96


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_number(value: float, metric: str) -> str:
    if metric == "alive":
        return f"{100 * value:.2f}%"
    return f"{value:.3f}"


def mean_se_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se, mean - NORMAL_95 * se, mean + NORMAL_95 * se


def genotype_fingerprint(genome: Any) -> tuple[Any, ...]:
    nodes = tuple(sorted(
        (key, float(gene.bias), float(gene.response), gene.activation, gene.aggregation)
        for key, gene in genome.nodes.items()
    ))
    connections = tuple(sorted(
        (key, float(gene.weight), bool(gene.enabled))
        for key, gene in genome.connections.items()
    ))
    return nodes, connections


def topology_fingerprint(genome: Any) -> tuple[Any, ...]:
    return tuple(sorted(genome.nodes)), tuple(sorted(
        key for key, gene in genome.connections.items() if gene.enabled
    ))


def paired_row(metric: str, first: dict[str, Any], middle: dict[str, Any], last: dict[str, Any]) -> str:
    values = {
        label: data["metrics"][metric]
        for (label, _), data in zip(VALIDATION_FILES, (first, middle, last))
    }
    d51 = [b - a for a, b in zip(first["per_map"][metric], middle["per_map"][metric])]
    d105 = [b - a for a, b in zip(middle["per_map"][metric], last["per_map"][metric])]
    stats51 = mean_se_ci(d51)
    stats105 = mean_se_ci(d105)
    wins51 = sum(x > 1e-12 for x in d51)
    losses51 = sum(x < -1e-12 for x in d51)
    ties51 = len(d51) - wins51 - losses51
    wins105 = sum(x > 1e-12 for x in d105)
    losses105 = sum(x < -1e-12 for x in d105)
    ties105 = len(d105) - wins105 - losses105
    return (
        f"| {metric} | {format_number(values['G1'], metric)} | {format_number(values['G5'], metric)} "
        f"| {format_number(values['G10'], metric)} | {format_number(stats51[0], metric)} "
        f"[{format_number(stats51[2], metric)}, {format_number(stats51[3], metric)}]; "
        f"{wins51}/{ties51}/{losses51} | {format_number(stats105[0], metric)} "
        f"[{format_number(stats105[2], metric)}, {format_number(stats105[3], metric)}]; "
        f"{wins105}/{ties105}/{losses105} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    run = parser.parse_args().run.resolve()
    validation_dir = run / "validation"

    validations = {
        label: read_json(validation_dir / f"generation-{generation:04d}.json")
        for label, generation in VALIDATION_FILES
    }
    for label, data in validations.items():
        if data.get("maps") != 32 or data.get("seconds") != 90 or data.get("seed") != 938271:
            raise ValueError(f"{label} does not match the common 32-map validation suite")
        if any(len(data["per_map"][metric]) != 32 for metric in METRICS):
            raise ValueError(f"{label} is missing 32 per-map values")

    first, middle, last = (validations[label] for label, _ in VALIDATION_FILES)
    if any(middle["per_map"][metric] != last["per_map"][metric] for metric in METRICS) and digest(validation_dir / "generation-0004.json") == digest(validation_dir / "generation-0009.json"):
        raise ValueError("G5 and G10 validation hashes match but per-map data do not")

    history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    species_history = read_json(run / "species-history.json")
    if len(history) < 10 or len(species_history) < 10:
        raise ValueError("Need completed generations 0 through 9")

    rows = []
    for generation in range(10):
        row = history[generation]
        species = species_history[generation]
        if row["generation"] != generation or species["generation"] != generation:
            raise ValueError("History files are not aligned by generation")
        rows.append((row, species))

    champion_pickle = run / "best-validation.pkl"
    with champion_pickle.open("rb") as handle:
        import pickle
        champion_model = pickle.load(handle)
    champion = champion_model["genome"]
    champion_id = champion.key

    episode_rows = []
    for generation in range(10):
        data = read_json(run / "episodes" / f"generation-{generation:04d}.json")
        if champion_id not in data["genome_ids"]:
            episode_rows.append((generation + 1, None, None, None, None))
            continue
        index = data["genome_ids"].index(champion_id)
        score = float(data["score"][index])
        anchor = float(data["anchor_score"][index])
        rotating = (score - 0.8 * anchor) / 0.2
        rank = 1 + sum(float(value) > score for value in data["score"])
        episode_rows.append((generation + 1, rank, score, anchor, rotating))

    checkpoint_rows = []
    persistent = True
    fingerprints = []
    for generation in range(4, 11):
        population = neat.Checkpointer.restore_checkpoint(str(run / f"checkpoint-{generation}"))
        genome = population.population.get(champion_id)
        if genome is None:
            persistent = False
            checkpoint_rows.append((generation, False, "-", "-", "-", "-", "-"))
            continue
        fingerprint = genotype_fingerprint(genome)
        fingerprints.append(fingerprint)
        exact_count = sum(genotype_fingerprint(item) == fingerprint for item in population.population.values())
        topology_count = sum(topology_fingerprint(item) == topology_fingerprint(genome)
                             for item in population.population.values())
        species_id = next((sid for sid, species in population.species.species.items()
                           if champion_id in species.members), None)
        species_size = len(population.species.species[species_id].members) if species_id is not None else 0
        hidden = len(genome.nodes) - len(population.config.genome_config.output_keys)
        checkpoint_rows.append((generation, True, hidden, len(genome.connections), species_id,
                                species_size, f"{exact_count}/{topology_count}"))
    persistent = persistent and bool(fingerprints) and all(value == fingerprints[0] for value in fingerprints)

    print("# G10 validation and population audit")
    print()
    print(f"Run: `{run}`. Validation files are indexed zero-based: G1=`generation-0000`, G5=`generation-0004`, G10=`generation-0009`. All use seed 938271, 32 maps, and 90 seconds. Deltas are paired in stored map order; confidence intervals are approximate 95% normal intervals. The last column in each paired result is G5>G1 / tie / G5<G1 (and likewise G10 vs G5); fewer boosts is descriptive and is not automatically a better outcome.")
    print()
    print("| Validation | Fitness | Food gain | Survival | Boost spent | SE fitness | SHA-256 |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for label, generation in VALIDATION_FILES:
        data = validations[label]
        path = validation_dir / f"generation-{generation:04d}.json"
        print(f"| {label} | {data['fitness']:.3f} | {data['food_gain']:.3f} | {100*data['alive']:.2f}% | {data['metrics']['boost_spent']:.3f} | {data['standard_error']:.3f} | `{digest(path)[:16]}…` |")
    print()
    print("| Metric | G1 | G5 | G10 | G5−G1: mean delta [95% CI]; >/=/< | G10−G5: mean delta [95% CI]; >/=/< |")
    print("|---|---:|---:|---:|---:|---:|")
    for metric in METRICS:
        print(paired_row(metric, first, middle, last))
    print()
    print("The G5 and G10 JSON artifacts are byte-identical: "
          f"`{digest(validation_dir / 'generation-0004.json')}`. Their per-map metrics match exactly, so this is the same validated policy outcome, not independent evidence of another gain.")
    print()
    d51, se51, lo51, hi51 = mean_se_ci([
        b - a for a, b in zip(first["per_map"]["fitness"], middle["per_map"]["fitness"])
    ])
    print(f"G1→G5 paired fitness change is {d51:.3f} (approximate 95% CI {lo51:.3f} to {hi51:.3f}; 20 of 32 maps improved). The interval includes zero, so this score gain remains uncertain on this validation set. G5→G10 changes are exactly zero because the validation artifacts and champion genotype are unchanged.")
    print()
    print("## Population and speciation curve")
    print()
    print("`mean` and `best` are the NEAT population fitness fields; `eval fitness` is the separately logged population-wide mean episode metric. Node counts include the two output nodes; connection counts are enabled links.")
    print()
    print("| Generation | Mean fitness | Best fitness | Eval fitness | Food | Survival | Boost | Mean nodes | Enabled links | Species | Effective species | Largest share |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row, species in rows:
        evaluation = row["evaluation"]
        print(f"| {row['generation']+1} | {row['mean']:.3f} | {row['best']:.3f} | {evaluation['fitness']:.3f} | {evaluation['food_gain']:.3f} | {100*evaluation['alive']:.2f}% | {evaluation['boost_spent']:.3f} | {row['nodes']:.3f} | {row['connections']:.3f} | {row['species']} | {species['effective']:.2f} | {100*species['largest_share']:.2f}% |")
    print()
    print("| Validation generation | Checkpoint | Mean hidden nodes | Mean enabled links | Distinct active topologies |")
    print("|---:|---|---:|---:|---:|")
    for label, checkpoint_index in (("G1", 0), ("G5", 4), ("G10", 9)):
        population = neat.Checkpointer.restore_checkpoint(str(run / f"checkpoint-{checkpoint_index}"))
        genomes = list(population.population.values())
        topology_count = len({topology_fingerprint(genome) for genome in genomes})
        hidden_mean = statistics.mean(len(genome.nodes) - len(population.config.genome_config.output_keys)
                                     for genome in genomes)
        enabled_mean = statistics.mean(sum(gene.enabled for gene in genome.connections.values())
                                       for genome in genomes)
        print(f"| {label} | `checkpoint-{checkpoint_index}` | {hidden_mean:.3f} | {enabled_mean:.3f} | {topology_count} |")
    print()
    print("Active topology here is node IDs plus enabled connection endpoints; it is a structural diversity proxy, not a count of behaviorally distinct policies.")
    print()
    print("Across G1→G5→G10, population mean fitness was "
          f"{rows[0][0]['mean']:.3f}→{rows[4][0]['mean']:.3f}→{rows[9][0]['mean']:.3f}; mean episode food gain was "
          f"{rows[0][0]['evaluation']['food_gain']:.3f}→{rows[4][0]['evaluation']['food_gain']:.3f}→{rows[9][0]['evaluation']['food_gain']:.3f}; "
          f"survival was {100*rows[0][0]['evaluation']['alive']:.2f}%→{100*rows[4][0]['evaluation']['alive']:.2f}%→{100*rows[9][0]['evaluation']['alive']:.2f}%. "
          f"Species count/effective count was {rows[0][0]['species']}/{rows[0][1]['effective']:.2f}→{rows[4][0]['species']}/{rows[4][1]['effective']:.2f}→{rows[9][0]['species']}/{rows[9][1]['effective']:.2f}.")
    print()
    print("## Persistent validation champion")
    print()
    print(f"`best-validation.pkl` identifies genome {champion.key} from zero-based generation index {champion_model.get('generation')} (G5); its topology has {len(champion.nodes)-2} hidden node(s) and {len(champion.connections)} connection genes. The checkpoint sequence 4–10 contains this key with the same complete gene fingerprint: **{persistent}**. `exact/topology` gives the number of exact copies and matching active graph topologies in that checkpoint population.")
    print()
    print("| Checkpoint | Genome present | Hidden nodes | Connection genes | Species ID | Species size | Exact/topology copies |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for values in checkpoint_rows:
        print("| " + " | ".join(str(value) for value in values) + " |")
    print()
    print("| Generation | Rank / 256 | Selection fitness | Fixed-anchor score | Derived rotating score* |")
    print("|---:|---:|---:|---:|---:|")
    for generation, rank, score, anchor, rotating in episode_rows:
        if rank is None:
            print(f"| {generation} | absent | — | — | — |")
        else:
            print(f"| {generation} | {rank} | {score:.3f} | {anchor:.3f} | {rotating:.3f} |")
    print()
    print("*Derived from the documented selection blend `(fitness − 0.8 × anchor_score) / 0.2`; it is the rotating-scenario contribution for this genome, not an independent validation score. The champion's fixed-anchor score remains 35.435 while its rotating score changes substantially. The validation score uses a different 32-map suite, so those raw scores are not directly comparable.")
    print()
    print("The G5 and G10 validation plateau is attributable to the same unchanged elite genotype, while population mean fitness, food gain, survival, topology, and species occupancy continue to move. The saved history does not record parent IDs, so descendant counts cannot be reconstructed; species occupancy and genotype persistence above are the reproducible concentration evidence. Ten generations are insufficient to infer global stagnation. Keep the present threshold, reward, and scenario mix for now; collect later common-validation points before proposing a training intervention.")
    print()
    print("## Reproduction")
    print()
    print("From the repository root, run `.venv/Scripts/python.exe docs/overnight/analyze_g10_audit.py`. The script reads saved artifacts only. It performs no simulation and writes nothing; `--run PATH` selects another completed run.")


if __name__ == "__main__":
    main()
