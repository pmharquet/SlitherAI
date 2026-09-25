# Audit de validation G25–G40

## Comparaison appariée

G25 désigne l’index 24 (checkpoint 25, génome gagnant 4107); G40, l’index 39 (checkpoint 40, génome gagnant 7080). Les validations utilisent la même graine `938271`, les 32 mêmes cartes et 90 s par carte. Les deltas sont calculés carte par carte, `G40 − G25`; l’erreur-type est l’écart-type des 32 différences divisé par `√32`, et les IC t bilatéraux à 95 % utilisent `df=31`, `t=2,0395`. Pour les résultats binaires, l’IC t apparié est indicatif; les nombres de cartes discordantes sont aussi donnés.

| Mesure | G25 → G40 | Delta apparié (SE; IC 95 %) | Cartes en faveur de G40 / G25 / égales |
|---|---:|---:|---:|
| Fitness | 21,263 → 24,749 | +3,485 (3,328; [−3,303; +10,274]) | 22 / 10 / 0 |
| Nourriture gagnée | 31,997 → 36,690 | +4,693 (3,874; [−3,207; +12,594]) | 21 / 11 / 0 |
| Vivant à 90 s | 17/32 → 20/32 | +9,38 pp (8,23; [−7,41; +26,16]) | 5 / 2 / 25 |
| Mort par bord | 10/32 → 6/32 | −12,50 pp (5,94; [−24,61; −0,39]) | 4 moins / 0 plus / 28 égales |
| Mort par collision | 5/32 → 6/32 | +3,13 pp (8,38; [−13,97; +20,22]) | 4 plus / 3 moins / 25 égales |

La moyenne fitness progresse de 3,49 points et G40 est meilleur sur 22 cartes. L’IC apparié couvre zéro; ces données donnent un signal favorable, pas une preuve précise d’un gain généralisable. La dispersion entre cartes (SE de la moyenne de validation) est `2,503` pour G25 et `2,666` pour G40; l’incertitude de leur différence appariée est plus grande (`SE 3,328`). La nourriture suit le même sens, avec un IC couvrant également zéro. Le décompte des morts par bord baisse de quatre cartes et celui des collisions augmente d’une; ces petits décomptes ne suffisent pas à établir un changement fiable de cause de décès. L’IC t nominal de bord frôle zéro, mais un test exact apparié n’aurait que quatre discordances toutes favorables à G40 (`p` bilatéral `0,125`).

## Sélection et génomes

La moyenne fitness de sélection enregistrée dans `history.jsonl` monte de `10,058` en G25 à `14,091` en G40; le meilleur score de génération passe de `36,992` à `43,302`. La moyenne de fitness des épisodes passe de `10,218` à `14,004`, la nourriture moyenne de `24,605` à `29,838` et la survie moyenne de `19,22 %` à `24,77 %`. Ce sont des mesures descriptives de deux populations évaluées avec un jeu tournant différent; elles ne remplacent pas le test apparié sur validation. Les deux checkpoints ont 38 espèces. Le gagnant G25 a 4 nœuds et 115 connexions (95 actives); G40 a 6 nœuds et 121 connexions (97 actives). G40 est un nouveau génotype, pas une répétition de G25.

| Génération / gagnant | Score de sélection | Espèce / taille | Nœuds | Connexions totales / actives | `_genome_hash` |
|---|---:|---:|---:|---:|---|
| G25 / ID 4107 | 36,992 | 31 / 10 | 4 | 115 / 95 | `1101711c875b87308d7db7ec397316957aeae3b2f84c29215809e6eb212ce7a9` |
| G40 / ID 7080 | 43,302 | 17 / 9 | 6 | 121 / 97 | `b26f3f2356fb0de910d3f6a44aa1ae7b30135e3853e17ca35a61bb206b97697c` |

L’argmax des scores d’épisodes identifie les gagnants, puis leurs génotypes ont été vérifiés dans les checkpoints 25 et 40. `best-validation.json` et le payload courant `best-validation.pkl` désignent maintenant la génération index 39, ID 7080, capteur `legacy-v1`; ses nœuds et connexions correspondent champ par champ au checkpoint 40 et à `best-validation-network.json`. L’empreinte utilise [`_genome_hash`](../../slitherai/evaluate_holdout.py), sur les attributs de nœuds et les attributs de connexions, y compris état activé et innovation.

## Lecture et limites

G40 est le nouveau meilleur score sur cette validation fixe et les moyennes de population observées progressent entre G25 et G40. La différence fitness appariée demeure incertaine sur 32 cartes, et choisir le meilleur parmi plusieurs checkpoints validés sur la même graine crée un biais de sélection. Ces résultats soutiennent la poursuite de l’observation, pas un réglage NEAT ou simulateur à partir de cette seule validation. Le holdout indépendant reste nécessaire pour estimer la généralisation.

## Provenance

Les fichiers de validation sont [G25](../../runs/20260924-163112-852097/validation/generation-0024.json) (SHA-256 `b48deb420a45e55cb3cdd9c0d230392ad0e143b82f85d955d9d19aad3963057b`) et [G40](../../runs/20260924-163112-852097/validation/generation-0039.json) (SHA-256 `70de11e8b60269c0c1b1b7e13367f74d0d75fc28ec4325a208a9a7ea06764a6d`). Les épisodes source sont `episodes/generation-0024.json` (SHA-256 `e916ef1cd47e5c0eeea1fe2a4865668e422f9563d04e71c131da8ccb604c0864`) et `episodes/generation-0039.json` (SHA-256 `e6168c9fb87c2f157078f10897a1e2a0a43fad7c981220737ac01707651e2c13`). Les checkpoints correspondants ont les SHA-256 `b24ff774359493c47261014a90a617e2473844cb02e95c053b5f7813b84be47c` et `d804e5f5772fc8526d3f5246c2ca41f20076f27a468f7361b43ed4967e221471`. Le payload `best-validation.pkl` est `15a22eb7948c820b901dde5232f73b82343fff9c3c73de5c9f4f813f1d88d9dd`; l’export réseau est `5aed60e090b007b84b5d0766383c18473443fe227df34ff02ba0462795ba5c0e`. `settings.json` est inchangé (SHA-256 `e8d7f1245a5ee5644e28606b8442797cbf3e24091949139917ddd576ef8f401b`).
