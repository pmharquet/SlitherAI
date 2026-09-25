# Bilan de l'entraînement nocturne — 25 septembre 2026

## Modèle retenu

Le candidat retenu avant l'épreuve indépendante est le champion **G45** (génération interne 44, génome **8270**), issu du run `runs/20260924-163112-852097`. Sa copie figée est [`best-validation-selected-20260925-073046.pkl`](../../runs/20260924-163112-852097/analysis/holdout-final/best-validation-selected-20260925-073046.pkl), SHA-256 `dac0e449ccc1803ab77b0d5169c295832b8ae39cded54344a648dde12b91d00d`. L'[export du réseau](../../runs/20260924-163112-852097/analysis/holdout-final/g45-best-validation-network.json) et les [métadonnées de validation](../../runs/20260924-163112-852097/analysis/holdout-final/g45-best-validation.json) ont également été figés. Son génotype a été vérifié contre le checkpoint 45 et l'export réseau : hash de génome `2482ca6e125e5a563cae50566100b4eb8add48344bf65e7254e2b820a5f84a4a`. Le réseau conserve le contrat **530 entrées, 2 sorties** et capteur `legacy-v1`; ses actions restent entièrement produites par NEAT.

Ce choix vient de la validation fixe existante, **avant** de lire l'épreuve indépendante. Les validations G1, G5, G15, G25, G40 et G45 donnaient respectivement **9,451; 14,069; 18,698; 21,263; 24,749; 27,438** points moyens sur les mêmes 32 cartes de sélection. Les comparaisons appariées [G25–G40](g25-g40-audit.md) et [G40–G45](g40-g45-audit.md) ont des intervalles à 95 % couvrant zéro : la hausse observée reste imprécise, et la sélection parmi plusieurs checkpoints sur cette même validation peut l'optimiser artificiellement.

## Épreuve finale indépendante

La suite réservée a été exécutée **une seule fois** le 25 septembre à 07:30–07:36 Paris : graine `741852963`, **64 cartes × 90 secondes**, mêmes paramètres de simulateur et adversaires de référence, CUDA. Le [résultat complet par carte](../../runs/20260924-163112-852097/analysis/holdout-final/holdout-741852963-64maps-20260925T053628Z.md) et le [JSON de provenance](../../runs/20260924-163112-852097/analysis/holdout-final/holdout-741852963-64maps-20260925T053628Z.json) contiennent les scores, différences appariées, incertitudes, configurations et hashes runtime.

| Politique | Fitness moyen ± erreur-type | Survie à 90 s | Nourriture moyenne |
|---|---:|---:|---:|
| Modèle initial G1 | 4,049 ± 1,774 | 8/64 | 16,911 |
| Champion G15 | 21,242 ± 2,286 | 36/64 | 32,956 |
| **Champion G45 retenu** | **24,905 ± 2,107** | **43/64** | **36,317** |
| Contrôleur heuristique de référence | 50,624 ± 3,175 | 36/64 | 139,897 |
| Contrôleur circulaire de référence | 2,138 ± 0,666 | 56/64 | 1,869 |

L'écart apparié **G45 − modèle initial** vaut **+20,856**, intervalle bootstrap à 95 % **[+15,429; +26,094]**. L'écart **G45 − G15** vaut **+3,664**, intervalle **[−1,432; +8,602]** : le point estimé favorise G45, mais cette suite ne tranche pas nettement entre ces deux champions. Le contrôleur heuristique obtient **+25,719** points face à G45, intervalle **[+18,205; +33,039]**. Il sert de repère et n'a jamais contrôlé les actions du modèle appris. Le cercle survit souvent mais collecte peu de nourriture; la fitness de croissance pénalise cette stratégie.

Ces intervalles décrivent les différences entre les **64 cartes de cette unique graine**. Ils ne prouvent pas les performances sur d'autres graines, d'autres adversaires ou le jeu d'origine. La suite finale ne doit plus servir aux réglages du modèle.

## Travail réalisé et décisions

- Entraînement NEAT CUDA repris depuis checkpoint 3, puis depuis checkpoint 17 après la disparition simultanée du service et des processus à 23:16. Les checkpoints et champions ont été préservés; la référence a continué avec les mêmes réglages `growth-v2`, `common-reference-v2`, 256 génomes, 64 cartes, 16 vers et 5 parties par génome. La cible reste **50 générations**. À **08:00 Paris**, le processus d'entraînement PID `6820` était actif, `pause=false`, `stop=false`, génération **G47 en cours** (index 46), **1 024/1 280 épisodes** traités, dernière génération achevée **G46** (index 45), dernier checkpoint complet **46**. Il est laissé en marche pour atteindre sa cible sans perte de génération. Les deux rappels nocturnes ont été désactivés à 08:00.
- L'[audit initial](initial-audit.md), l'[inventaire des paramètres](parameter-map.md), les audits de validations et l'outil `slitherai.overnight_audit` ont séparé score de sélection, score par partie et validation. Les **38 espèces** persistantes ne démontrent pas un effondrement. Le seuil de compatibilité a atteint **4,0**; la [reclassification CPU G28](threshold-cap-g28.md) aux seuils 4, 5, 6 et 8 donnait toujours 38 niches avec les représentants existants. Le plancher de reproduction 4→2 ne libérait pas de budget de descendants à G3. Aucun réglage de spéciation n'a donc été changé sur une interprétation cosmétique.
- Une erreur de provenance a été corrigée : sérialiser le config NEAT avançait le compteur de neurones cachés. Le patch rend les snapshots de champion/checkpoint purs; les tests de reprise ciblés ont été vérifiés avant le redémarrage. Cette correction préserve la reproductibilité future, sans prétendre accroître la fitness.
- Le mode capteur `export-v1` a été essayé dans un run distinct sur cinq générations; sa validation finale n'a pas montré de gain établi par rapport au legacy. Deux mesures bornées suggèrent que `sensor_chunk=8` pourrait accélérer la simulation (~1,44–1,50× sur les scènes chronométrées), mais la parité complète des trajectoires CUDA n'est pas établie. Le réglage **4** du run de référence n'a pas été modifié. Un pilote parallèle préflighté a été refusé par l'approbation automatique avant lancement; aucun contournement ni second essai n'a été fait.
- L'outil de comparaison finale a passé un smoke CPU puis CUDA avec des graines non réservées. Les tests ciblés pertinents ont passé lors des revues; le [préflight](holdout-preflight.md) décrit les identités et garde-fous. Tous les commits restent **locaux**, sans push. Aucun comportement spécialisé n'a été programmé dans le contrôleur du candidat.

## Limites et suite utile

Les deux exports JSONL du jeu d'origine annoncés dans `C:\Users\88mat\Downloads` étaient absents lors des vérifications nocturnes. La fidélité chiffrée du simulateur au jeu ne peut donc pas être confirmée; ses règles de densité, géométrie, croissance et adversaires sont encore des approximations documentées. Le score heuristique supérieur montre aussi qu'il reste beaucoup à apprendre dans ce simulateur.

Laisser la référence atteindre son checkpoint 50, puis comparer tout nouveau candidat sur **une autre suite indépendante** définie avant l'analyse; la graine finale de cette nuit est consommée. Quand les exports du jeu seront disponibles, calibrer en priorité le simulateur et les observations, puis lancer des expériences versionnées une hypothèse à la fois. Les actions du ver appris doivent continuer à venir exclusivement de son réseau.
