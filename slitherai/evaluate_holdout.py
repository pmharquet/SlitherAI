"""Controlled offline comparison of frozen SlitherAI policies.

Every reported result is freshly simulated with ``evaluation.play_episode``.
Pickle inputs are trusted local run artifacts and must not come from untrusted
sources. This module never starts training or edits a run's configuration.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimConfig
from .evaluation import play_episode, protocol_settings
from .rewards import REWARD_VERSION
from .schema import VERSION, contract


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {type(exc).__name__}") from exc


def _schema_ok(value: Any) -> bool:
    return value == VERSION or value == contract()


def _validate_genome_config(genome: Any, neat_config: Any) -> None:
    gc = neat_config.genome_config
    if gc.num_inputs != contract()["inputs"] or gc.num_outputs != len(contract()["outputs"]):
        raise ValueError("Model NEAT input/output dimensions do not match the active observation schema")
    # Constructing the phenotype catches unsupported activation/aggregation genes
    # before the more expensive simulation starts.
    import neat
    neat.nn.RecurrentNetwork.create(genome, neat_config)


def _genome_hash(genome: Any) -> str:
    nodes = []
    for key, node in sorted(genome.nodes.items()):
        nodes.append({"id": key, "bias": node.bias, "response": node.response,
                      "activation": node.activation, "aggregation": node.aggregation})
    connections = []
    for key, gene in sorted(genome.connections.items()):
        connections.append({"source": key[0], "target": key[1], "weight": gene.weight,
                            "enabled": gene.enabled, "innovation": getattr(gene, "innovation", None)})
    return _canonical_hash({"nodes": nodes, "connections": connections})


def _checkpoint_champion(run: Path, generation: int) -> dict[str, Any]:
    if generation < 0:
        raise ValueError("Generation must be zero or greater")
    episode_path = run / "episodes" / f"generation-{generation:04d}.json"
    episode = _read_json(episode_path)
    if not isinstance(episode, dict):
        raise ValueError(f"Episode record is not an object for generation {generation}")
    ids, scores = episode.get("genome_ids"), episode.get("score")
    if episode.get("generation") != generation or not isinstance(ids, list) or not ids or not isinstance(scores, list):
        raise ValueError(f"Episode record is incomplete for generation {generation}")
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.ndim != 1 or len(scores_array) != len(ids) or not np.isfinite(scores_array).all():
        raise ValueError(f"Episode scores are invalid for generation {generation}")
    winner_id = ids[int(scores_array.argmax())]
    checkpoint_path = run / f"checkpoint-{generation}"
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint is missing: {checkpoint_path}")
    # This is intentionally an explicit trusted-local pickle load: the checkpoint
    # stores NEAT genes/configuration, while the JSON score array identifies the
    # champion without trusting a cached fitness as a comparison result.
    from .analyze_training import load_checkpoint
    saved_generation, neat_config, population, _, _ = load_checkpoint(checkpoint_path)
    if int(saved_generation) != generation:
        raise ValueError(f"Checkpoint generation mismatch for {checkpoint_path}")
    genome = population.get(winner_id)
    if genome is None:
        raise ValueError(f"Champion genome {winner_id!r} is absent from {checkpoint_path}")
    _validate_genome_config(genome, neat_config)
    return {
        "name": f"generation-{generation}", "genome": genome, "neat_config": neat_config,
        "source": {
            "kind": "checkpoint_episode_champion", "path": str(checkpoint_path.resolve()),
            "episode_path": str(episode_path.resolve()), "generation": generation,
            "genome_id": winner_id, "selection_score_used_only_to_select": float(scores_array.max()),
            "checkpoint_sha256": _file_hash(checkpoint_path),
            "episode_sha256": _file_hash(episode_path), "genome_sha256": _genome_hash(genome),
        },
    }


def _payload_model(label: str, path: Path) -> dict[str, Any]:
    source_path = path.resolve()
    try:
        payload = pickle.loads(source_path.read_bytes())
    except Exception as exc:
        raise ValueError(f"Cannot load trusted local model payload {source_path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict) or not {"genome", "config", "schema"}.issubset(payload):
        raise ValueError(f"Model payload must contain genome, config, and schema: {source_path}")
    if not _schema_ok(payload["schema"]):
        raise ValueError(f"Model schema is incompatible with {VERSION}: {source_path}")
    genome, neat_config = payload["genome"], payload["config"]
    _validate_genome_config(genome, neat_config)
    generation = payload.get("generation")
    return {
        "name": label, "genome": genome, "neat_config": neat_config,
        "source": {"kind": "trusted_model_payload", "path": str(source_path),
                   "generation": generation, "genome_id": genome.key,
                   "payload_sha256": _file_hash(source_path), "genome_sha256": _genome_hash(genome)},
    }


def _load_simulation_config(run: Path, comparison_path: Path | None, maps: int) -> tuple[SimConfig, dict[str, Any]]:
    settings = _read_json(run / "settings.json")
    run_schema = _read_json(run / "schema.json")
    if not isinstance(settings, dict):
        raise ValueError("Run settings must be a JSON object")
    if not _schema_ok(run_schema):
        raise ValueError(f"Run observation schema is incompatible with {VERSION}")
    saved_protocol = settings.get("protocol")
    saved_config = settings.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Run settings have no simulation config")
    if comparison_path is None:
        if saved_protocol != protocol_settings():
            raise ValueError("Saved protocol differs; provide --comparison-config to rescore every policy under one protocol")
        sim_config = SimConfig.from_dict(saved_config)
        if sim_config.reward_version != REWARD_VERSION:
            raise ValueError("Saved reward version differs; provide --comparison-config to rescore every policy")
        config_source = "saved_run_settings"
    else:
        comparison = _read_json(comparison_path)
        values = comparison.get("config") if isinstance(comparison, dict) else None
        expected_fields = {field.name for field in dataclasses.fields(SimConfig)}
        if not isinstance(values, dict) or set(values) != expected_fields:
            raise ValueError("Comparison config must include every SimConfig field, including reward_version")
        if comparison.get("protocol") != protocol_settings():
            raise ValueError("Comparison config protocol must exactly match the active evaluation protocol")
        sim_config = SimConfig.from_dict(values)
        config_source = str(comparison_path.resolve())
    sim_config.maps = maps
    sim_config.validate()
    metadata = {
        "source": config_source, "config": dataclasses.asdict(sim_config),
        "config_sha256": _canonical_hash(dataclasses.asdict(sim_config)),
        "protocol": protocol_settings(), "protocol_sha256": _canonical_hash(protocol_settings()),
        "reward_version": sim_config.reward_version,
        "reward_version_sha256": _canonical_hash(sim_config.reward_version),
        "schema_version": VERSION, "schema_sha256": _canonical_hash(contract()),
        "all_policies_freshly_rescored": True,
    }
    return sim_config, metadata


def _summary(values: list[Any]) -> dict[str, float | int | None]:
    numbers = np.asarray(values, dtype=np.float64)
    if numbers.size == 0 or not np.isfinite(numbers).all():
        return {"n": int(numbers.size), "mean": None, "standard_error": None}
    se = float(numbers.std(ddof=1) / math.sqrt(len(numbers))) if len(numbers) > 1 else None
    return {"n": int(len(numbers)), "mean": float(numbers.mean()), "standard_error": se}


def _paired_delta(a: list[Any], b: list[Any], seed: int, label: str) -> dict[str, Any]:
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Paired outcomes must be finite vectors of the same length")
    delta = left - right
    result: dict[str, Any] = {"n": int(len(delta)), "mean_difference": float(delta.mean()),
                              "paired_standard_error": float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta)>1 else None}
    if len(delta) > 1:
        # Map bootstrap describes uncertainty within this fixed scenario suite;
        # it does not turn the suite into a sample of the real game distribution.
        token = hashlib.sha256(f"{seed}:{label}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
        indices = rng.integers(0, len(delta), size=(2000, len(delta)))
        means = delta[indices].mean(axis=1)
        result["bootstrap_95_percentile_interval"] = [float(v) for v in np.quantile(means, [.025, .975])]
    else:
        result["bootstrap_95_percentile_interval"] = None
    return result


def _evaluate(model: dict[str, Any], sim_config: SimConfig, seed: int, seconds: float, device: str) -> dict[str, Any]:
    maps = sim_config.maps
    focal = np.arange(maps, dtype=np.int64) * 7 % sim_config.worms
    genome = model.get("genome")
    rows = play_episode([genome] * maps if genome is not None else None,
                        model.get("neat_config"), sim_config, device, seed, focal, seconds,
                        shared_random=False, policy=model.get("policy", "neat"))
    return {key: {"summary": _summary(values), "per_map": values} for key, values in rows.items()}


def run_holdout(run: Path, candidate_generations: list[int], candidate_payloads: list[tuple[str, Path]],
                seed: int, maps: int, seconds: float, device: str = "cpu",
                comparison_path: Path | None = None) -> dict[str, Any]:
    if maps < 1 or maps > 512 or seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("maps must be between 1 and 512 and seconds must be finite and positive")
    run = run.resolve()
    sim_config, config_meta = _load_simulation_config(run, comparison_path, maps)
    steps = round(seconds / sim_config.dt)
    if not math.isclose(seconds, steps * sim_config.dt, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("seconds must align to the saved simulation timestep")
    models = [_checkpoint_champion(run, 0)]
    models[0]["name"] = "initial"
    for generation in candidate_generations:
        model = _checkpoint_champion(run, generation)
        if generation == 0:
            model["name"] = "candidate-generation-0"
        models.append(model)
    for label, path in candidate_payloads:
        model = _payload_model(label, path)
        try:
            same_run = Path(model["source"]["path"]).is_relative_to(run)
        except ValueError:
            same_run = False
        if not same_run and comparison_path is None:
            raise ValueError("External model payloads require --comparison-config so every policy is rescored explicitly")
        models.append(model)
    labels = [model["name"] for model in models] + ["heuristic", "circle"]
    if len(labels) != len(set(labels)):
        raise ValueError("Policy labels must be unique; candidate payload labels cannot collide")
    models.extend(({"name": "heuristic", "policy": "heuristic"},
                   {"name": "circle", "policy": "circle"}))

    evaluated = []
    for model in models:
        result = _evaluate(model, sim_config, seed, seconds, device)
        evaluated.append({"name": model["name"], "source": model.get("source", {"kind": "fixed_builtin_policy"}),
                          "metrics": result})
    comparisons = []
    metrics = list(evaluated[0]["metrics"])
    for index, challenger in enumerate(evaluated):
        for comparator in evaluated[:index]:
            paired = {key: _paired_delta(challenger["metrics"][key]["per_map"],
                                          comparator["metrics"][key]["per_map"], seed,
                                          f"{challenger['name']}:{comparator['name']}:{key}")
                      for key in metrics}
            comparisons.append({"challenger": challenger["name"], "comparator": comparator["name"],
                                "metrics": paired})
    return {
        "schema": "slitherai-offline-holdout-v1", "run": str(run), "seed": seed,
        "maps": maps, "seconds": seconds, "worms": sim_config.worms, "focal_slots": "map_index*7 % worms",
        "opponents": "evaluation.heuristic in every non-focal slot",
        "device": device, "comparison": config_meta, "models": evaluated,
        "paired_comparisons": comparisons,
        "uncertainty_note": "Paired map standard errors and fixed-seed bootstrap intervals describe variation within this declared suite; they do not establish generalization to new seeds, opponents, or the source game.",
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Offline policy comparison", "",
             f"Run: `{result['run']}`  ",
             f"Suite: seed `{result['seed']}`, {result['maps']} maps, {result['seconds']:g}s, device `{result['device']}`  ",
             f"Config SHA-256: `{result['comparison']['config_sha256']}`  ",
             f"Protocol `{result['comparison']['protocol']['version']}` SHA-256: `{result['comparison']['protocol_sha256']}`  ",
             f"Reward `{result['comparison']['reward_version']}` SHA-256: `{result['comparison']['reward_version_sha256']}`", "",
             "All metrics below were freshly simulated. Historical episode scores were used only to identify generation champions.", "",
             "| Policy | Generation | Fitness mean ± SE | Food gain | Survival | Age (s) | Kills |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in result["models"]:
        metrics = model["metrics"]
        generation = model["source"].get("generation", "—")
        def fmt(key: str, suffix: str = "") -> str:
            value = metrics.get(key, {}).get("summary", {})
            mean, se = value.get("mean"), value.get("standard_error")
            if mean is None: return "—"
            return f"{mean:.3f} ± {se:.3f}" if se is not None and key == "fitness" else f"{mean:.3f}{suffix}"
        lines.append(f"| {model['name']} | {generation} | {fmt('fitness')} | {fmt('food_gain')} | {fmt('alive')} | {fmt('age')} | {fmt('kills')} |")
    lines += ["", "## Paired fitness differences", "", "Positive values favor the challenger.", "",
              "| Challenger | Comparator | Mean difference | Paired SE | 95% bootstrap interval |", "|---|---|---:|---:|---:|"]
    for row in result["paired_comparisons"]:
        fit = row["metrics"].get("fitness", {})
        interval = fit.get("bootstrap_95_percentile_interval")
        interval_text = f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "—"
        se = fit.get("paired_standard_error")
        se_text = f"{se:.3f}" if se is not None else "—"
        lines.append(f"| {row['challenger']} | {row['comparator']} | {fit['mean_difference']:.3f} | {se_text} | {interval_text} |")
    lines += ["", result["uncertainty_note"], "",
              "## Provenance", "", "| Policy | Source | Genome ID | Genome SHA-256 |", "|---|---|---:|---|"]
    for model in result["models"]:
        source = model["source"]
        lines.append(f"| {model['name']} | `{source.get('path', source.get('kind'))}` | {source.get('genome_id', '—')} | `{source.get('genome_sha256', '—')}` |")
    return "\n".join(lines) + "\n"


def _candidate_payload(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Candidate payload must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Candidate payload must be LABEL=PATH")
    return label.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore frozen policies on one controlled offline scenario suite")
    parser.add_argument("--run", required=True, type=Path, help="Trusted local training run directory")
    parser.add_argument("--candidate-generation", action="append", type=int, default=[],
                        help="Add champion reconstructed from this generation's episode score and checkpoint (repeatable)")
    parser.add_argument("--candidate-payload", action="append", type=_candidate_payload, default=[],
                        help="Add a trusted saved model payload as LABEL=PATH (repeatable)")
    parser.add_argument("--seed", required=True, type=int, help="Explicit scenario seed")
    parser.add_argument("--maps", required=True, type=int, help="Explicit independent arena count")
    parser.add_argument("--seconds", required=True, type=float, help="Explicit simulated seconds per arena")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--comparison-config", type=Path,
                        help="Explicit full config+protocol JSON; all policies are freshly rescored with it")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN/analysis/holdout")
    args = parser.parse_args()
    if not args.candidate_generation and not args.candidate_payload:
        parser.error("provide at least one --candidate-generation or --candidate-payload")
    result = run_holdout(args.run, args.candidate_generation, args.candidate_payload,
                         args.seed, args.maps, args.seconds, args.device, args.comparison_config)
    out_dir = (args.output_dir or args.run / "analysis" / "holdout").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"holdout-{args.seed}-{args.maps}maps-{stamp}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path),
                      "policies": [model["name"] for model in result["models"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
