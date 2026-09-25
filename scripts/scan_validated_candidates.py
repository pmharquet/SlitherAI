"""Preselect and freshly validate the strongest recorded generation candidates.

The historical scores choose which five genotypes to load from each declared
generation. Every unique genotype is then freshly evaluated with the project's
fixed validation scenario. Pickle inputs are trusted local run artifacts.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slitherai.analyze_training import load_checkpoint
from slitherai.config import SimConfig
from slitherai.evaluate_holdout import (
    _file_hash,
    _genome_hash,
    _load_simulation_config,
    _payload_model,
    _read_json,
    _validate_genome_config,
)
from slitherai.evaluation import fixed_validation
from slitherai.schema import sensor_version_from_schema
from slitherai.train import _config_pickle_snapshot


GENERATIONS = (0, 4, 9)
TOP_PER_GENERATION = 5
VALIDATION_SEED = 938271
VALIDATION_MAPS = 32
VALIDATION_SECONDS = 90.0
G45_CHECKPOINT_INDEX = 44
G45_GENOME_ID = 8270


def select_top_five(episode: Any, generation: int) -> list[dict[str, Any]]:
    """Select exactly five recorded scores, with source order breaking ties."""
    if not isinstance(episode, dict) or episode.get("generation") != generation:
        raise ValueError(f"Episode generation record does not match G{generation}")
    genome_ids, scores = episode.get("genome_ids"), episode.get("score")
    if not isinstance(genome_ids, list) or not isinstance(scores, list) or len(genome_ids) != len(scores):
        raise ValueError(f"Episode IDs/scores are incomplete for G{generation}")
    if len(scores) < TOP_PER_GENERATION:
        raise ValueError(f"G{generation} has fewer than {TOP_PER_GENERATION} scored genomes")
    if any(not isinstance(genome_id, int) or isinstance(genome_id, bool) for genome_id in genome_ids):
        raise ValueError(f"G{generation} contains a non-integer genome ID")
    if len(set(genome_ids)) != len(genome_ids):
        raise ValueError(f"G{generation} contains duplicate genome IDs")
    score_values = np.asarray(scores, dtype=np.float64)
    if score_values.ndim != 1 or not np.isfinite(score_values).all():
        raise ValueError(f"G{generation} episode scores are invalid")
    ranked_indices = sorted(range(len(scores)), key=lambda index: (-score_values[index], index))
    return [
        {"rank": rank, "episode_index": index, "genome_id": genome_ids[index],
         "historical_score_for_selection_only": float(score_values[index])}
        for rank, index in enumerate(ranked_indices[:TOP_PER_GENERATION], start=1)
    ]


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse exact canonical genotypes while retaining every source alias."""
    unique: list[dict[str, Any]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        genome_hash = _genome_hash(candidate["genome"])
        candidate["genome_sha256"] = genome_hash
        existing = by_hash.get(genome_hash)
        if existing is None:
            candidate.setdefault("aliases", [])
            by_hash[genome_hash] = candidate
            unique.append(candidate)
        else:
            existing["aliases"].append({
                "name": candidate["name"],
                "source": candidate["source"],
                "historical_score_for_selection_only": candidate.get("historical_score_for_selection_only"),
            })
    return unique


def _load_generation_candidates(run: Path, generation: int, sensor_version: str) -> list[dict[str, Any]]:
    episode_path = run / "episodes" / f"generation-{generation:04d}.json"
    episode = _read_json(episode_path)
    selections = select_top_five(episode, generation)
    checkpoint_path = run / f"checkpoint-{generation}"
    if not checkpoint_path.is_file():
        raise ValueError(f"Missing checkpoint for G{generation}: {checkpoint_path}")
    try:
        saved_generation, neat_config, population, _, _ = load_checkpoint(checkpoint_path)
    except Exception as exc:
        raise ValueError(f"Cannot load trusted local checkpoint {checkpoint_path}: {type(exc).__name__}") from exc
    if int(saved_generation) != generation:
        raise ValueError(f"Checkpoint generation mismatch for G{generation}: {saved_generation}")

    checkpoint_hash = _file_hash(checkpoint_path)
    episode_hash = _file_hash(episode_path)
    models = []
    for selection in selections:
        genome_id = selection["genome_id"]
        genome = population.get(genome_id)
        if genome is None:
            raise ValueError(f"G{generation} genome {genome_id} from the episode is absent from its checkpoint")
        _validate_genome_config(genome, neat_config)
        models.append({
            "name": f"G{generation}-rank{selection['rank']}-ID{genome_id}",
            "genome": genome,
            "neat_config": neat_config,
            "source": {
                "kind": "generation_episode_top5",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "episode_path": str(episode_path.resolve()),
                "episode_sha256": episode_hash,
                "generation": generation,
                "rank_within_generation": selection["rank"],
                "episode_index": selection["episode_index"],
                "genome_id": genome_id,
                "sensor_version": sensor_version,
            },
            "historical_score_for_selection_only": selection["historical_score_for_selection_only"],
        })
    return models


def _load_witness(path: Path, expected_sensor_version: str) -> dict[str, Any]:
    model = _payload_model("G45-frozen-witness", path)
    if model["source"].get("generation") != G45_CHECKPOINT_INDEX:
        raise ValueError("Frozen G45 witness must identify zero-based generation 44")
    if model["source"].get("genome_id") != G45_GENOME_ID:
        raise ValueError("Frozen G45 witness must identify genome 8270")
    if model["source"].get("sensor_version") != expected_sensor_version:
        raise ValueError("G45 witness sensor schema differs from the run")
    model["source"]["kind"] = "frozen_g45_payload_witness"
    model["source"]["display_generation"] = 45
    return model


def _json_safe(value: Any) -> Any:
    """Convert NumPy values in fixed_validation's per-map output to JSON types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _prepare_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError(f"Output directory must be new or empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise ValueError(f"Refusing to overwrite existing output: {path}") from exc


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Validated candidate scan", "",
        f"Run: `{result['run']}`  ",
        f"Validation: seed `{result['seed']}`, {result['maps']} maps × {result['seconds']:g}s, device `{result['device']}`  ",
        f"Config SHA-256: `{result['comparison']['config_sha256']}`  ",
        f"Schema: `{result['comparison']['schema_version']}`; reward: `{result['comparison']['reward_version']}`  ",
        "Historical episode scores were used only to choose the five candidates in each generation. "
        "All fitness values below were freshly measured on the fixed validation suite.", "",
        "| Model | Source | Historical rank/score | Validation fitness ± SE | Food gain | Survival | Age (s) | Kills | Genome SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["models"]:
        validation = row["validation"]
        source = row["source"]
        rank = source.get("rank_within_generation")
        historical = row.get("historical_score_for_selection_only")
        selection = f"{rank} / {historical:.4f}" if rank is not None and historical is not None else "witness"
        generation_label = (f"G{source['display_generation']} (checkpoint {source['generation']})"
                            if source.get("display_generation") is not None
                            else f"G{source.get('generation', '—')}")
        age = validation.get("metrics", {}).get("age")
        age_text = f"{age:.3f}" if age is not None else "—"
        lines.append(
            f"| {row['name']} | {generation_label} | {selection} | "
            f"{validation['fitness']:.4f} ± {validation['standard_error']:.4f} | "
            f"{validation['food_gain']:.3f} | {validation['alive']:.3f} | {age_text} | "
            f"{validation['kills']:.3f} | `{row['genome_sha256']}` |"
        )
        for alias in row.get("aliases", []):
            lines.append(f"| ↳ {alias['name']} (same genotype) | alias | — | — | — | — | — | — | `{row['genome_sha256']}` |")
    best = result["best_candidate"]
    lines += [
        "", f"Best validation candidate: **{best['name']}** (fitness {best['fitness']:.4f}).",
        f"Frozen payload: `{best['payload_path']}`  ",
        f"Payload SHA-256: `{best['payload_sha256']}`  ",
        f"Comparison config: `{result['comparison_config_path']}`  ",
        "",
        "The fixed validation suite supports prescreening only; final performance should be measured once on a separately reserved suite.",
    ]
    return "\n".join(lines) + "\n"


def run_scan(run: Path, witness_payload: Path, output_dir: Path, device: str) -> dict[str, Any]:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    run = run.resolve()
    witness_payload = witness_payload.resolve()
    out = _prepare_output_dir(output_dir)
    sim_config, comparison = _load_simulation_config(run, None, VALIDATION_MAPS)
    steps = round(VALIDATION_SECONDS / sim_config.dt)
    if not math.isclose(VALIDATION_SECONDS, steps * sim_config.dt, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("90-second validation is not aligned to the run timestep")
    run_schema = _read_json(run / "schema.json")
    run_sensor = sensor_version_from_schema(run_schema)
    if run_sensor is None or run_sensor != sim_config.sensor_version:
        raise ValueError("Run schema and saved simulation sensor_version are unsupported or inconsistent")

    candidates: list[dict[str, Any]] = []
    for generation in GENERATIONS:
        candidates.extend(_load_generation_candidates(run, generation, run_sensor))
    candidates.append(_load_witness(witness_payload, run_sensor))
    candidates = deduplicate_candidates(candidates)

    rows = []
    for candidate in candidates:
        validation = fixed_validation(candidate["genome"], candidate["neat_config"], sim_config,
                                      device, seconds=VALIDATION_SECONDS, maps=VALIDATION_MAPS)
        rows.append({
            "name": candidate["name"],
            "source": candidate["source"],
            "aliases": candidate["aliases"],
            "genome_sha256": candidate["genome_sha256"],
            "historical_score_for_selection_only": candidate.get("historical_score_for_selection_only"),
            "validation": validation,
            "_genome": candidate["genome"],
            "_neat_config": candidate["neat_config"],
        })

    best = max(rows, key=lambda row: row["validation"]["fitness"])
    config_path = out / "comparison-config.json"
    config_payload = {"config": dataclasses.asdict(sim_config), "protocol": comparison["protocol"]}
    _write_new(config_path, (json.dumps(config_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))

    best_payload_path = out / "best-candidate.pkl"
    payload = {
        "genome": best["_genome"],
        "config": _config_pickle_snapshot(best["_neat_config"]),
        "schema": comparison["schema"],
        "generation": best["source"].get("generation"),
        "selection": {
            "method": "fixed-validation-fitness",
            "seed": VALIDATION_SEED,
            "maps": VALIDATION_MAPS,
            "seconds": VALIDATION_SECONDS,
            "fitness": best["validation"]["fitness"],
            "genome_sha256": best["genome_sha256"],
            "source": best["source"],
            "aliases": best["aliases"],
        },
    }
    _write_new(best_payload_path, pickle.dumps(payload))
    best_payload_hash = _file_hash(best_payload_path)

    serializable_rows = [_json_safe({key: value for key, value in row.items() if not key.startswith("_")})
                         for row in rows]
    result = {
        "schema": "slitherai-validated-candidate-scan-v1",
        "run": str(run),
        "seed": VALIDATION_SEED,
        "maps": VALIDATION_MAPS,
        "seconds": VALIDATION_SECONDS,
        "device": device,
        "focal_slots": "map_index*7 % worms",
        "opponents": "evaluation.heuristic in every non-focal slot",
        "candidate_rule": {"generations": list(GENERATIONS), "top_per_generation": TOP_PER_GENERATION,
                           "rank_order": "descending recorded score; original episode order breaks ties",
                           "recorded_scores_used_only_for_candidate_selection": True},
        "comparison": comparison,
        "runtime": {
            "scan_script_sha256": _file_hash(Path(__file__).resolve()),
            "evaluation_py_sha256": _file_hash(ROOT / "slitherai" / "evaluation.py"),
        },
        "models": serializable_rows,
        "best_candidate": {
            "name": best["name"],
            "fitness": best["validation"]["fitness"],
            "genome_sha256": best["genome_sha256"],
            "payload_path": str(best_payload_path.resolve()),
            "payload_sha256": best_payload_hash,
        },
        "comparison_config_path": str(config_path.resolve()),
        "comparison_config_sha256": _file_hash(config_path),
        "uncertainty_note": "The fixed validation suite is a prescreen, not an independent estimate of generalization; candidate selection on it can overfit to its seed and scenarios.",
    }
    json_path = out / "scan.json"
    markdown_path = out / "scan.md"
    _write_new(json_path, (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    _write_new(markdown_path, render_markdown(result).encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select recorded top-five genotypes and freshly score them on fixed validation")
    parser.add_argument("--run", required=True, type=Path, help="Trusted local run with episodes and checkpoints")
    parser.add_argument("--witness-payload", required=True, type=Path, help="Trusted frozen G45 model payload")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty directory for scan and frozen best payload")
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"), help="Explicit evaluation device")
    args = parser.parse_args()
    result = run_scan(args.run, args.witness_payload, args.output_dir, args.device)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "unique_models": len(result["models"]),
                      "best_candidate": result["best_candidate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
