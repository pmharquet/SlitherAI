# Pilote parallèle `sensor_chunk=8` depuis `checkpoint-21`

## État et préflight CPU

Le checkpoint retenu est `runs/20260924-163112-852097/checkpoint-21`. Il est cohérent avec l’épisode et la ligne d’historique de génération 20, qui documentent la dernière population évaluée avant sa création. Paramètres source vérifiés : 64 cartes, 16 vers, population 256, 90 s, seed d’entraînement 1, validations toutes les 5 générations, mode `legacy-v1`, récompense `growth-v2`, protocole `common-reference-v2`, chunk 4. La cible source est génération 50 ; il restait 29 générations à partir du checkpoint 21.

Un préflight CPU a appelé `initialize_from_checkpoint()` sur ce checkpoint dans un répertoire temporaire hors de `runs/`. Les 256 génomes ont été restaurés, validés, puis préparés en mémoire pour un nouveau run à génération 0 avec `sensor_chunk=8` explicite. Le protocole, la récompense, le schéma et les paramètres physiques passent les contrôles de compatibilité. Aucun simulateur, GPU, entraînement, processus ou fichier source n’a été touché.

| Source | SHA-256 |
|---|---|
| `checkpoint-21` (290 321 octets) | `901f8106678ac924f9230e286df8fb79bdb08c2bfcf4850cc9d4791f0d315e6b` |
| `episodes/generation-0020.json` | `cab1fcc51e77e1b219c6467c1bc0eb8cd42fa82c1f35f81ae891677b8c82e076` |
| `settings.json` | `e8d7f1245a5ee5644e28606b8442797cbf3e24091949139917ddd576ef8f401b` |
| `schema.json` | `70da0c788bb275cdadcad4ffe723c572e68356f1f3ac467890d49782a4eaddf7` |

Ces empreintes identifient le snapshot préparé. Avant lancement, la commande ci-dessous refuse de démarrer si le checkpoint, les settings ou le schéma ont changé. Le warm-start inscrit aussi ses hashes et l’identité du checkpoint dans `initialization.json` du nouveau run.

## Commande préparée, non exécutée

Cette variante laisse la référence active et démarre cinq générations dans un nouveau run depuis exactement le checkpoint 21. Le destinataire n’est pas créé par PowerShell : `Trainer` doit le recevoir vide. Les sorties stdout/stderr sont placées sous `runs/_chunk8-pilot-logs/`, hors du nouveau run, pour ne pas faire échouer la vérification de destination vide.

À lancer seulement après revue et décision GPU :

```powershell
$root = 'C:\Docker\SlitherAI'
$run = Join-Path $root 'runs\20260924-163112-852097'
$checkpoint = Join-Path $run 'checkpoint-21'
$settings = Join-Path $run 'settings.json'
$schema = Join-Path $run 'schema.json'
$expected = @{
  $checkpoint = '901f8106678ac924f9230e286df8fb79bdb08c2bfcf4850cc9d4791f0d315e6b'
  $settings = 'e8d7f1245a5ee5644e28606b8442797cbf3e24091949139917ddd576ef8f401b'
  $schema = '70da0c788bb275cdadcad4ffe723c572e68356f1f3ac467890d49782a4eaddf7'
}
foreach ($path in @($checkpoint, $settings, $schema)) {
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
  if ($actual -ne $expected[$path]) { throw "Source hash changed: $path" }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$pilot = Join-Path $root "runs\chunk8-pilot-cp21-$stamp"
if (Test-Path -LiteralPath $pilot) { throw "Destination already exists: $pilot" }
$logDir = Join-Path $root 'runs\_chunk8-pilot-logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "chunk8-cp21-$stamp.stdout.log"
$stderr = Join-Path $logDir "chunk8-cp21-$stamp.stderr.log"
$python = Join-Path $root '.venv\Scripts\python.exe'
$trainArgs = @(
  '-m', 'slitherai.train', '--run', $pilot,
  '--initialize-from', $checkpoint,
  '--maps', '64', '--worms', '16', '--foods', '1024', '--body-points', '96',
  '--arena-radius', '2400', '--population', '256', '--generations', '5',
  '--seconds', '90', '--seed', '1', '--validation-every', '5',
  '--sensor-version', 'legacy-v1', '--sensor-chunk', '8', '--device', 'cuda'
)
$process = Start-Process -FilePath $python -ArgumentList $trainArgs `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process | Select-Object Id, ProcessName, StartTime
```

`Start-Process` n’a pas été appelé. Aucun contrôle API, fichier `control.json`, `last-run.json`, checkpoint ou paramètre du run source n’a été modifié pendant la préparation. Contrôler après décision le PID renvoyé, `runs/<nouveau-run>/status.json`, `settings.json`, `initialization.json` et les deux journaux.

## Coût GPU et lecture des résultats

Les cinq dernières durées source G16–G20 sont 2 029,74, 1 153,59, 1 044,73, 1 188,41 et 1 058,41 s ; leur médiane est 1 153,59 s, soit environ 96 min pour cinq générations au chunk 4. Un benchmark antérieur a mesuré une partie de 64 cartes × 16 vers × 60 s en 36,231 s au chunk 4 et 24,216 s au chunk 8 (ratio 1,496). C’était une seule mesure chronométrée, ordre fixe, sans répétition/warm-up ; l’extrapolation indicative donnerait environ 64 min pour cinq générations chunk 8 en exécution isolée, avant validations et calibration des contrôleurs. Ce n’est pas une prévision validée. En parallèle, les deux entraînements se partageront le même GPU et le ralentissement de la référence n’est pas mesuré. Le pilote ferait 6 400 simulations d’entraînement (5 × 256 génomes × 5 parties), en plus des validations.

Le warm-start conserve les génomes, gènes, compteurs d’innovations et état RNG du checkpoint, mais remet fitness, ancres, comportements mémorisés, espèces, âges et historique de stagnation à zéro. Les espèces sont reformées en génération zéro. Ainsi, G1 (index 0) mesure les génomes importés après rescoring frais ; G5 (index 4) valide la branche après cinq générations scorées. Avec `validation_every=5`, les deux validations utilisent les 32 cartes fixes, 90 s et seed 938271. Le premier passage calcule aussi les deux contrôleurs de référence absents du nouveau run. Ces scores sont comparables à la validation source au niveau du protocole, mais restent réutilisés pour la sélection et ne constituent pas un test indépendant. Le cinquième scénario d’entraînement du pilote repart de l’index de génération 0 : cette branche n’est pas une continuation exacte de la séquence des scénarios du run source.

Le dashboard restera sur la référence : `/api/start` sait créer un run neuf ou reprendre strictement, mais n’accepte pas `--initialize-from`. La commande lance donc le pilote directement, sans modifier `runs/last-run.json`; l’interface ne le suivra ni ne le contrôlera. Surveiller les fichiers status du pilote et les logs ci-dessus. Le pilote n’est pas déployé dans le service par cette préparation.

Interpréter une différence de validation comme un signal limité : le warm-start réinitialise la spéciation et les fitness, et la seule variation de chunk déjà mesurée en CUDA n’a pas démontré la parité exacte des trajectoires. Aucune conclusion sur l’effet causal du chunk ou sur le transfert au jeu réel ne découle de cinq générations parallèles.

## Tentative de lancement du 25 septembre 2026

Après autorisation, les trois empreintes source ont été revérifiées et concordaient. L’API indiquait le run de référence actif en phase `training`; les processus trainer observés appartenaient à ce run. La commande préparée a ensuite été soumise à `exec_command`, qui l’a refusée avant son exécution : `CreateProcess: Rejected(...): blocked by policy` (la commande refusée contenait `Start-Process`). Aucun autre mécanisme de lancement n’a été essayé.

État lecture seule observé à 00:43:41 Europe/Paris : le run de référence était toujours actif, génération 21/50, 768/1280 épisodes, environ 37,3 s par épisode ; `nvidia-smi` indiquait 353/8188 MiB et 38 % d’utilisation. Aucun répertoire `chunk8-pilot-cp21-*` n’existait. Le pilote n’a donc pas démarré ; seule la référence avançait. La commande reste à lancer par un contexte qui autorise explicitement `Start-Process`, après nouvelle vérification des hashes et du service.
