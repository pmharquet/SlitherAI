# Diagnostic du comportement près des bordures

`scripts/diagnose_border_behavior.py` rejoue un ou deux génomes sauvegardés sur les cartes déterministes de validation et produit un rapport JSON. Il utilise le `settings.json` de chaque run, le chargeur de payloads du projet, `WorldBatch`, `BatchedNetwork` et l’heuristique existante pour les adversaires. L’option `--include-heuristic` rejoue aussi ce contrôleur comme ver focal pour disposer d’un témoin. CPU est le périphérique par défaut; CUDA est disponible avec `--device cuda`. Le périphérique utilisé figure dans le rapport. Le script n’écrit pas les runs sauf si `--output` pointe explicitement vers un fichier de run.

Depuis la racine du dépôt, après la fin de l’entraînement GPU :

```powershell
.\.venv\Scripts\python.exe -m scripts.diagnose_border_behavior `
  --candidate runs\run-a runs\run-a\best-validation.pkl `
  --candidate runs\run-b runs\run-b\best-validation.pkl `
  --device cuda --output docs\border-replay.json
```

Chaque `--candidate` prend le dossier contenant `settings.json`, puis le payload du génome. Donnez une ou deux paires. Les paramètres de simulation sauvegardés doivent être identiques pour comparer les mêmes cartes. Par défaut, le script rejoue les 32 cartes, pendant 90 secondes, avec la graine `938271` et la même attribution de siège que `fixed_validation`.

Pour vérifier rapidement le script sur deux cartes et deux pas (`dt=0.1`) :

```powershell
.\.venv\Scripts\python.exe -m scripts.diagnose_border_behavior `
  --candidate runs\smoke runs\smoke\best-validation.pkl `
  --maps 2 --seconds 0.2 --device cpu
```

Le rapport décrit explicitement la provenance et les empreintes SHA-256 des settings et payloads. Pour le ver focal, il donne la survie, le score, la nourriture, les décès, la part des ticks vivants à moins de 5 %, 10 % et 20 % du rayon d’arène de la bordure, la nourriture gagnée près/loin selon sa position au début du tick, et le changement de cap moyen. Il indique aussi la première entrée dans la bande de 10 %, la proportion de temps dans cette bande durant chaque tiers de partie et le nombre de tours autour du centre réalisés dans cette bande. La distance de bordure est `rayon d’arène - distance du centre à la tête - rayon du ver`, ramenée au rayon de l’arène. Les adversaires et la mécanique sont ceux du projet; le script ne définit aucune stratégie supplémentaire.
