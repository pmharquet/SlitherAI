# Checkpoint warm-start reset policy

## Scope

`--initialize-from runs/<source>/checkpoint-N` imports the checkpoint's live genomes into a **new empty run**. It is separate from strict `--resume`; the two options are mutually exclusive. The source must be a trusted local checkpoint made by this project. The initializer reads source metadata and checkpoint bytes but never writes to the source directory.

The destination's `settings.json` and `initialization.json` record source path, checkpoint generation and SHA-256, source settings/config/schema hashes, protocol hash, source and destination sensor modes, imported genome IDs, destination config/schema hashes, and the reset policy. A changed sensor mode is accepted only when explicitly selected with `--sensor-version`. Without that flag, the source sensor mode is inherited.

## State disposition

| State | Warm-start decision | Reason |
| --- | --- | --- |
| Live population IDs and node/connection genes | Preserve exactly; clear only cached `behavior` | This is the learned policy population. The check before the first evaluation verifies IDs and all node/connection attributes. |
| Per-genome `fitness` and `anchor_fitness` | Reset to `None` | Scores came from the source run's episodes and cannot rank the new run. Every genome must be evaluated before reproduction. |
| Source NEAT config | Preserve | Mutation rates, crossover, topology parameters, and compatible input/output keys must continue consistently with the imported genes. |
| Innovation tracker global counter | Preserve; reject if it trails a live connection innovation | New structural mutations must not reuse an existing innovation number. Clear only the generation-local deduplication map because generation zero is a new reproduction epoch. |
| Node-key counter | Preserve from the checkpoint | New hidden nodes must be allocated above the imported topology. |
| Genome ID allocator | Restore using NEAT-Python's checkpoint behavior: next ID is one greater than the largest imported live ID | NEAT checkpoints do not store the reproduction object or its historical allocator. This is collision-safe within the fresh destination; the source's extinct genome ID history is unavailable. |
| Species and stagnation | Rebuild at generation zero with a fresh species set and stagnation object | Ages, fitness histories, anchor medians, protection/removal flags and cached adjusted fitness belong to the old objective. The adaptive species threshold is recalibrated against the unchanged imported population. Species IDs restart in the new run. |
| Python random state | Preserve the checkpoint state after species setup | Species implementations may consume randomness while clustering. The importer captures/restores the checkpoint state so evolution resumes from that RNG stream. The destination seed continues to govern NumPy scenario generation. |
| Best genome and source artifacts | Clear `best_genome`; copy no history, episodes, validation, baselines, settings, or champion files | The new run must form all selection and validation evidence under its own settings. |

The importer permits `maps` and `sensor_chunk` to differ because they affect batching/work allocation, not physical rules. It checks every other simulation field, including body physics, view dimensions, reward version, and timestep. The 530 input and two output keys, order, activation, and aggregation are checked against this trainer. Protocol and source schema must be recognized and current.

## Review points and limits

- A warm-start across `legacy-v1` and `export-v1` preserves exactly the same initial policy genes but changes the values arriving on the self-body channel. The initial population must be evaluated afresh; source champion ranks and cached scores are not carried over.
- The adaptive species threshold is recalibrated on the imported genotypes, while mutation/crossover settings and innovation numbering remain intact. Species IDs and stagnation history start over. This favors a clean new objective while keeping the learned topologies.
- The checkpoint format does not store NEAT's historical genome-ID allocator or ancestry archive. The restored allocator starts above the largest currently live genome key, which prevents collisions in the new population but cannot preserve IDs of extinct source genomes. Runs have separate histories, so those historical IDs are not used to join records across runs.
- CPU tests verify preserved genes, counters, source bytes, reset state, destination safety, RNG restoration, and fresh innovation numbering. They stub the evaluation loop. No GPU or sensor-mode training run is performed by these tests.
- Before training from an `export-v1` warm-start, review the new run's provenance, run the prescribed fresh validation/holdout, and complete CUDA parity/performance checks. Synthetic JS/Python geometry parity alone is not a transfer result.
