# Préflight de l’épreuve finale réservée

## Smokes CPU

Un premier smoke précédent a validé le CLI avec le modèle initial, un payload et les contrôleurs, sur une seed non réservée, 1 carte × 0,1 s. Il n’est pas une mesure de politique.

Le préflight ciblé demandé a rescored les deux candidats prévus sur le run de référence, CPU seulement, avec seed non réservée `2123456`, 1 carte × 0,1 s :

```powershell
.\.venv\Scripts\python.exe -m slitherai.evaluate_holdout `
  --run runs\20260924-163112-852097 `
  --candidate-generation 14 `
  --candidate-payload "best-validation=runs\20260924-163112-852097\analysis\holdout-preflight-g15-g25-20260925-020829\best-validation-current-snapshot.pkl" `
  --seed 2123456 --maps 1 --seconds 0.1 --device cpu `
  --output-dir runs\20260924-163112-852097\analysis\holdout-preflight-g15-g25-20260925-020829
```

Le JSON a confirmé ces politiques et identités :

| Politique | Génération stockée | Génome | Origine |
|---|---:|---:|---|
| `initial` | 0 | 76 | champion de l’épisode initial |
| `generation-14` (G15) | 14 | 2867 | champion rescored depuis `checkpoint-14` et l’épisode 14 |
| `best-validation` (G25) | 24 | 4107 | payload de validation figé pour ce smoke |
| `heuristic` | — | — | contrôleur intégré |
| `circle` | — | — | contrôleur intégré |

Le snapshot de payload a le SHA-256 `cf33179dbc04b1431577c35f30c23c657f710d606c7c7c7bd48aa9f1c91681c5`. Les fichiers de résultat sont dans `runs/20260924-163112-852097/analysis/holdout-preflight-g15-g25-20260925-020829/`. Le smoke a vérifié le chargement, la compatibilité et les cinq entrées de politique ; 1 carte et 0,1 s ne mesurent ni la performance ni une validation.

## Smoke CUDA effectué

Une seconde smoke a exécuté les mêmes deux candidats, sur CUDA, avec une graine distincte non réservée (`2718281`), toujours 1 carte × 0,1 s :

```powershell
.\.venv\Scripts\python.exe -m slitherai.evaluate_holdout `
  --run runs\20260924-163112-852097 `
  --candidate-generation 14 `
  --candidate-payload "best-validation=runs\20260924-163112-852097\analysis\holdout-preflight-g15-g25-20260925-020829\best-validation-current-snapshot.pkl" `
  --seed 2718281 --maps 1 --seconds 0.1 --device cuda `
  --output-dir runs\20260924-163112-852097\analysis\holdout-gpu-smoke-20260925-043646
```

La sortie contient `initial` (ID 76), G15/ID2867, G25/ID4107 et les contrôleurs `heuristic` et `circle`. Le JSON consigne `device=cuda`, Python `3.12.6`, PyTorch `2.8.0+cu129` et 8 empreintes SHA-256 de fichiers runtime, toutes valides. La carte est une NVIDIA GeForce RTX 4060 Laptop GPU ; juste après le smoke, `nvidia-smi` lisait 353/8188 MiB et 29 % d’utilisation. L’API confirmait la référence active, génération 35/50, sur le run prévu. Le résultat JSON/Markdown est dans `analysis/holdout-gpu-smoke-20260925-043646/`. Il s’agit seulement d’un smoke de chargement et d’exécution CUDA, pas d’un score de politique.

## Candidats et règle de sélection

G15 (génération interne 14, génome 2867) est le candidat présélectionné depuis l’historique. L’autre candidat sera l’unique payload `best-validation.pkl` retenu au moment où la décision finale est prise. Au préflight ci-dessus, ce payload correspond à G25 (génération interne 24, génome 4107), choisi sur les validations existantes à seed `938271`, 32 cartes × 90 s (fitness 21,263 ± 2,503). Il peut évoluer si une validation ultérieure le remplace ; le choix doit alors être arrêté avant le holdout.

Après avoir arrêté le choix, copier ce payload dans un nom horodaté, comparer les hashes source et copie, puis l’évaluer uniquement depuis cette copie immuable. L’épreuve finale compare G15 et ce payload figé ; le CLI ajoute aussi le modèle initial et les contrôleurs `heuristic` et `circle`. Exécuter une seule invocation avec la seed réservée. Ne pas comparer un autre payload après avoir consulté le résultat.

## Commande finale préparée — ne pas exécuter avant le choix

La seed finale `741852963` n’a pas été utilisée dans les smokes. Quand la référence et la sélection sont prêtes, exécuter le bloc ci-dessous une seule fois. Il refuse un répertoire de sortie déjà existant ou un payload qui change pendant sa copie. Il recalcule les deux hash avant le lancement et conserve le payload figé avec le résultat.

```powershell
$ErrorActionPreference = 'Stop'
$run = 'runs\20260924-163112-852097'
$finalDir = Join-Path $run 'analysis\holdout-final'
if (Test-Path -LiteralPath $finalDir) { throw "Final output already exists: $finalDir" }
New-Item -ItemType Directory -Path $finalDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$candidate = Join-Path $finalDir "best-validation-selected-$stamp.pkl"
$source = Join-Path $run 'best-validation.pkl'
$sourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
Copy-Item -LiteralPath $source -Destination $candidate
$candidateHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
$sourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
if ($sourceHashBefore -ne $candidateHash -or $sourceHashBefore -ne $sourceHashAfter) {
  throw 'best-validation changed during snapshot; do not run the reserved holdout.'
}
"Frozen payload SHA-256: $candidateHash"

.\.venv\Scripts\python.exe -m slitherai.evaluate_holdout `
  --run $run `
  --candidate-generation 14 `
  --candidate-payload "best-validation=$candidate" `
  --seed 741852963 --maps 64 --seconds 90 --device cuda `
  --output-dir $finalDir
if ($LASTEXITCODE -ne 0) { throw "evaluate_holdout exited $LASTEXITCODE" }
```

Le JSON final enregistre les scores frais par carte, différences appariées, protocole, récompense, schéma, configuration, identité et hash de chaque candidat, et hashes du code évaluateur. Les résultats servent une seule fois ; aucun choix de gagnant ou réglage postérieur n’est reporté comme s’il avait été prédit par cette épreuve.
