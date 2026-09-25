# Audit de validation G40–G45

## Validation appariée

G40 est l’index 39 (checkpoint 40, gagnant de sélection ID `7080`); G45 est l’index 44 (checkpoint 45, ID `8270`). Les deux validations couvrent les mêmes 32 cartes, avec seed `938271` et 90 s par carte. Les deltas sont `G45 − G40`, calculés carte par carte; `SE = écart-type des différences / √32`, IC t bilatéral à 95 % avec `df=31`, `t=2,0395`. Les intervalles sur proportions binaires sont indicatifs.

| Mesure | G40 → G45 | Delta (SE; IC 95 %) | Cartes : G45 mieux / G40 mieux / égales |
|---|---:|---:|---:|
| Fitness | 24,749 → 27,438 | +2,689 (3,314; [−4,069; +9,447]) | 19 / 13 / 0 |
| Nourriture | 36,690 → 39,271 | +2,580 (3,696; [−4,957; +10,118]) | 19 / 13 / 0 |
| Vivant à 90 s | 20/32 → 23/32 | +9,38 pp (8,23; [−7,41; +26,16]) | 5 / 2 / 25 |
| Mort par bord | 6/32 → 6/32 | 0 pp | 0 / 0 / 32 |
| Mort par collision | 6/32 → 3/32 | −9,38 pp (8,23; [−26,16; +7,41]) | 5 moins / 2 plus / 25 égales |

La moyenne de fitness et la survie montent, mais l’IC de fitness appariée couvre zéro. L’erreur-type entre cartes de chaque moyenne est `2,666` en G40 et `2,590` en G45; le SE apparié (`3,314`) est plus pertinent pour leur différence. La nourriture s’améliore aussi au point estimé, avec un IC couvrant zéro. Les morts par bord restent à six cartes; les collisions diminuent de trois sans précision suffisante pour conclure à une baisse reproductible.

## Sélection, population et identité du champion

La moyenne de fitness de sélection de la population passe de `14,091` à `14,958` entre G40 et G45, tandis que le meilleur score de génération baisse de `43,302` à `38,860`. La moyenne de fitness des épisodes monte légèrement (`14,004` → `14,639`), la nourriture reste proche (`29,838` → `29,472`) et la survie populationnelle augmente (`24,77 %` → `29,77 %`). Ce sont des comparaisons descriptives : le jeu rotatif change, et le score de sélection ne mesure pas les 32 cartes fixes. Les deux checkpoints comptent 38 espèces et le seuil de compatibilité reste au plafond `4,0`. Les nœuds moyens montent de `3,98` à `4,39`; les connexions actives moyennes passent de `102,76` à `102,13`.

| Génération / ID | Score de sélection | Espèce / taille | Nœuds | Connexions / actives | `_genome_hash` |
|---|---:|---:|---:|---:|---|
| G40 / 7080 | 43,302 | 17 / 9 | 6 | 121 / 97 | `b26f3f2356fb0de910d3f6a44aa1ae7b30135e3853e17ca35a61bb206b97697c` |
| G45 / 8270 | 38,860 | 10 / 9 | 4 | 114 / 100 | `2482ca6e125e5a563cae50566100b4eb8add48344bf65e7254e2b820a5f84a4a` |

`best-validation.json` et `best-validation.pkl` pointent maintenant vers la génération index 44, ID `8270`, capteur `legacy-v1`. Le génotype du checkpoint 45, le payload pickle et `best-validation-network.json` correspondent exactement, champ par champ pour les nœuds et connexions. L’empreinte est calculée avec [`_genome_hash`](../../slitherai/evaluate_holdout.py). G45 devient le meilleur score de validation observé, mais l’écart sur 32 cartes ne suffit pas à établir un gain généralisable; les validations répétées sur cette seed exposent aussi au biais de sélection des checkpoints.

## Provenance

Validations : [G40](../../runs/20260924-163112-852097/validation/generation-0039.json), SHA-256 `70de11e8b60269c0c1b1b7e13367f74d0d75fc28ec4325a208a9a7ea06764a6d`; [G45](../../runs/20260924-163112-852097/validation/generation-0044.json), SHA-256 `c1c1ead70d45fe54031f46f26f2f165f9bd9bc9c83c06cda681dbfab0c52249e`. Épisodes : index 39 `e6168c9fb87c2f157078f10897a1e2a0a43fad7c981220737ac01707651e2c13`; index 44 `4e96980ef1f2831f6c9fcaeae3bcd8181447024ccd4da00642aeccd9cafc598f`. Checkpoints : 40 `d804e5f5772fc8526d3f5246c2ca41f20076f27a468f7361b43ed4967e221471`; 45 `dac0ebcfd76498c5940d7483f8d5c5d6e666d5f3bd3620dbda8ef7ced15a43f3`. Payload `best-validation.pkl`: `dac0e449ccc1803ab77b0d5169c295832b8ae39cded54344a648dde12b91d00d`; réseau JSON : `bd99f236724dfad199f816788e15b389590134715f7b47eda09a00e480d98597`. Les réglages proviennent de `settings.json`, SHA-256 `e8d7f1245a5ee5644e28606b8442797cbf3e24091949139917ddd576ef8f401b`.
