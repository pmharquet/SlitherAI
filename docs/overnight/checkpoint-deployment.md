# Corrective checkpoint deployment

Recorded 2026-09-24 (Europe/Paris). The trainer was stopped at a complete checkpoint boundary, then resumed from that checkpoint with the serialization-purity fix. The server was not restarted and no process was killed.

## Checkpoint and interrupted work

- Run: `runs/20260924-163112-852097`.
- Before stop, the newest stable artifact was `checkpoint-8`, 278,099 bytes, SHA-256 `8ba43e6849512aa780fe545212232f2988a210faa8d8c3411a139234490c4e11` (file time 2026-09-24 20:01:56 +02:00). The prior `checkpoint-7` hash was `52b7882469c6845a3199a5721bc00c1f46a70ec472c7ccbc1ed69827fdae8f2d`.
- The run history ended at row/generation 7. The status API called the next in-progress generation 8; when stopped it had recorded 64 of 1,280 episodes (5%) for that generation. That partial generation was discarded. No generation-8 history row or later checkpoint had been written.
- A graceful API stop returned `active=false`, `phase=stopped`, generation 8, target 50, and no error. The stop was issued only after confirming `checkpoint-8` was stable.

## Resume and verification

- At 2026-09-24 20:04:24 +02:00, the service API accepted `POST /api/start` with `{"resume":true,"device":"cuda","generations":42}` for the same run. The trainer command resumed from `checkpoint-8` with `--generations 42 --device cuda`.
- The run settings remained seed 1, population 256, maps 64, worms 16, foods 1,024, body points 96, arena radius 2,400, episode length 90 seconds, validation every 5 generations, `growth-v2` reward, and `common-reference-v2` evaluation (4 anchor games + 1 rotating game, anchor weight 0.8). The total target remained generation 50; 42 additional generations were requested from the checkpoint.
- API verification at 20:04:30 showed `active=true`, `phase=training`, generation 8 of 50, CUDA on the NVIDIA GeForce RTX 4060 Laptop GPU, and no error. At 20:05:59 it was still active and had advanced to 128/1,280 episodes in generation 8. `control.json` was reset to `pause=false`, `stop=false`, `arena=0`.
- One service-managed trainer job was present: launcher PID 20948 with its Python child PID 8028, parented by the unchanged server PID 1208 listening on port 8765. No duplicate trainer was found.
- The checkpoint hash remained unchanged after resume. The most recent committed history row was still generation 7 during the verification window; generation 8 was in progress.

## Provenance

- Training source at restart: commit `b88959574c45112061709e47d6d93e4197ae82cf`; `slitherai/train.py` SHA-256 `6366bdcad96075b260a5985b0ffc7744048f754abed164f0a6f666a105c74347`. This source includes the checkpoint-purity fix from commit `a712b2b53478979b0888a853ec90c8879baf6273`.
- The fix and its resume/serialization tests passed before deployment. No training configuration or checkpoint was edited by this deployment procedure.
- The only incomplete work intentionally lost was the 64/1,280 episodes from the interrupted generation-8 attempt. The resumed process began a fresh attempt from the intact `checkpoint-8` artifact.
