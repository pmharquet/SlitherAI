# Préflight de l’épreuve finale réservée

## Smoke CPU effectué

Commande très courte exécutée sur le run de référence, seed non réservée `314159`, une carte et `0,1` seconde simulée :

```powershell
.\.venv\Scripts\python.exe -m slitherai.evaluate_holdout `
  --run runs\20260924-163112-852097 `
  --candidate-payload best-validation=runs\20260924-163112-852097\best-validation.pkl `
  --seed 314159 --maps 1 --seconds 0.1 --device cpu `
  --output-dir runs\20260924-163112-852097\analysis\holdout-preflight-smoke
```

Résultat : environ 3,8 s, avec les politiques `initial`, `best-validation`, `heuristic` et `circle`. Le modèle initial a été reconstruit avec l’argmax de l’épisode 0000, puis retrouvé dans `checkpoint-0` (génération 0, génome 76). Le payload de validation a été chargé comme génération 14, génome 2867, SHA-256 `682fd88f3e4c1d8a7adaa57bdb2654c3d7dc9c709289fc043200394c42d70731`. Le smoke a utilisé Python 3.12.6, PyTorch 2.8.0+cu129 et NEAT-Python 1.1.0. Le JSON et le Markdown sont sous `analysis/holdout-preflight-smoke/` ; les settings et checkpoints du run n’ont pas été modifiés.

Ce smoke vérifie le chargement des sources, la compatibilité du schéma, la simulation CPU, les contrôleurs et l’écriture de la provenance. Une carte d’un pas ne mesure pas la performance des politiques et n’est pas une validation.

## Figer le candidat avant la graine réservée

La sélection doit être faite sur les validations existantes avant de lancer le test. Au snapshot du préflight, le meilleur payload du run de référence est `best-validation.pkl`, génération 14 / génome 2867, sélectionné sur 32 cartes et 90 s à la seed de validation `938271` (fitness moyenne `18,698 ± 3,017`). Si une validation ultérieure remplace ce champion, choisir le meilleur payload validé à ce moment-là avant le holdout.

Comme le run de référence peut encore écrire `best-validation.pkl`, copier le payload sélectionné dans un nom horodaté avant l’épreuve et calculer son SHA-256. L’entraîneur écrit d’abord un temporaire puis remplace le payload ; une fois la copie faite, l’évaluateur ne lira que ce fichier immuable. Son JSON consignera le chemin, la génération, l’ID de génome et le hash réellement scorés, ce qui permet de vérifier la sélection copiée même si la validation du run écrit ensuite un nouveau best. Faire cette copie après que le choix du modèle est arrêté ; ne pas comparer plusieurs snapshots sur le seed final.

## Commande finale préparée — à lancer une seule fois

La graine finale `741852963` n’a pas été utilisée par le préflight. Une fois l’entraînement terminé et le candidat choisi, exécuter ces commandes PowerShell. Elles copient le meilleur payload validé du run de référence vers un fichier horodaté, impriment son hash, puis comparent ce candidat au modèle initial et aux deux contrôleurs intégrés. L’évaluateur ne modifie ni entraînement ni configuration.

```powershell
$run = 'runs\20260924-163112-852097'
$finalDir = Join-Path $run 'analysis\holdout-final'
New-Item -ItemType Directory -Force -Path $finalDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$candidate = Join-Path $finalDir "best-validation-selected-$stamp.pkl"
Copy-Item -LiteralPath (Join-Path $run 'best-validation.pkl') -Destination $candidate
Get-FileHash -Algorithm SHA256 -LiteralPath $candidate

.\.venv\Scripts\python.exe -m slitherai.evaluate_holdout `
  --run $run `
  --candidate-payload "best-validation=$candidate" `
  --seed 741852963 --maps 64 --seconds 90 --device cuda `
  --output-dir $finalDir
```

Le JSON final conserve les mesures fraîches par carte, les écarts appariés, le protocole, la récompense, le schéma, le hash de configuration, l’identité/hash du candidat figé et les hash du code exécuté. La seed est réservée à cette invocation. Ne pas choisir un autre candidat après avoir consulté ce résultat sans signaler la sélection sur test ; ce même holdout ne fournirait alors plus une mesure indépendante du gagnant.
