# Priorités d’expérimentation après G1–G12

Lecture seule du run `runs/20260924-163112-852097`, de `parameter-map.md`, du trainer, des épisodes, du protocole et de `configs/neat.ini`. G1–G12 désigne les lignes d’historique d’indices 0–11. Aucun entraînement, replay GPU ou changement de paramètres n’a été lancé pour cette analyse.

## Ce que les données permettent de dire

- La moyenne de fitness de sélection passe de −17,93 à +2,92 entre G1 et G12, mais le champion de validation G5 reste identique en G10. Sur les 32 cartes de validation (graine 938271), G5−G1 vaut +4,62 points avec un intervalle apparié approximatif de −2,24 à +11,48 ; le signal de progression sur cet ensemble demeure incertain. G10 réutilise exactement les résultats G5.
- Le nombre d’espèces passe de 8 à 38 et reste à 38 depuis G4. L’audit G10 mesure une diversité effective de 37,14 et aucun génome dans les espèces minuscules. La cible de 8–12 n’est donc pas une preuve que 38 espèces nuisent à l’apprentissage.
- Le nombre moyen de nœuds, sorties comprises, passe de 2,00 à 2,94 ; cela représente environ 0,94 nœud caché par génome à G12. Le nombre de connexions reste près de 105. C’est une faible complexité moyenne, mais pas une preuve que la capacité limite la politique.
- Le signal d’ancrage et la sélection restent souvent proches. Une contre-factuelle exacte à partir des champs enregistrés `score` et `anchor_score`, avec le poids d’ancrage porté de 0,8 à 0,9, garde le meilleur génome de 11 cohortes sur 12 et conserve en moyenne 9 des 10 premiers. Ce test gratuit n’identifie pas de raison forte pour changer la pondération de sélection cette nuit.
- Il n’existe qu’une graine de validation distincte : 938271, 32 cartes, 90 s. La réutiliser est acceptable pour un filtre apparié de présélection, pas pour une preuve indépendante. La graine finale 741852963 reste réservée.

### Contre-factuelle exacte du plancher (CPU, G12)

`docs/overnight/diagnose_speciation.py` a rejoué uniquement `reproduce()` en mémoire pour `checkpoint-12`, en injectant les fitness de `episodes/generation-0012.json` et les mesures de `history.jsonl`. Au snapshot consulté, G13 était encore en cours : `checkpoint-12` et l’épisode G12 étaient la dernière paire complète. Aucun simulateur n’a tourné. Le checkpoint contient 256 génomes, mais seulement 76 valeurs `fitness`/`anchor_fitness` non nulles des élites de la génération précédente ; il ne suffit donc pas seul pour reconstruire la reproduction après l’évaluation G12. L’épisode sauvegardé fournit le score des 256 génomes et ses cinq scénarios.

| `min_species_size` | Espèces actives | Allocation par espèce | Au plancher effectif | Élites / nouveaux enfants | Somme top 4 / top 8 |
|---:|---:|---|---:|---:|---:|
| 4 (effectif 4) | 38 | 5:4, 6:18, 7:7, 8:4, 9:3, 10:2 | 0 | 76 / 180 | 38 / 71 |
| 2 (effectif 2, car `elitism=2`) | 38 | 4:1, 5:2, 6:19, 7:7, 8:4, 9:3, 10:2 | 0 | 76 / 180 | 38 / 71 |

Le changement effectif touche seulement deux espèces : l’ID 1 passe de 5 à 4 places et l’ID 5 de 5 à 6. Les parts top 4 et top 8 restent respectivement 14,84 % et 27,73 % ; la plus grande part reste 3,91 %. Le plancher 4 n’est actif dans aucune allocation G12 : `38×4=152` est une borne théorique, pas 152 places effectivement immobilisées. L’effet mesuré de 4→2 est minime sur cette reproduction ; `min_species_size` est donc rétrogradé au rang 3 et n’est pas un essai prioritaire cette nuit. Reproduction exacte : `.\.venv\Scripts\python.exe docs/overnight/diagnose_speciation.py --no-replay --allocation-generation 12`.

## Trois essais à classer

| Rang | Paramètre unique | A → B | Pourquoi ce test a un signal mesurable |
|---|---|---|---|
| 1 | `survival_threshold` | 0,30 → 0,45 | Avec les tailles G12, le pool de parents passe de 98 à 136 places au total et change dans 35 des 38 espèces. Les élites et les quotas de descendants restent inchangés. |
| 2 | `node_add_prob` | 0,15 → 0,25 | Davantage d’enfants tenteraient de scinder un lien en un neurone et deux connexions. La moyenne reste sous un nœud caché par génome à G12 ; l’essai mesure si un peu plus de structure améliore l’ancre et la validation. Il interagit avec `node_delete_prob=0,02`, `conn_add_prob=0,50`, le réseau récurrent et le coût d’activation. |
| 3 | `min_species_size` | 4 → 2 | La reproduction exacte G12 ne touche que deux allocations d’un slot, sans espèce au plancher, sans changement des parts top 4/top 8 et sans changement du nombre d’enfants générés. Le contraste est donc mesurable mais peu prometteur à court terme. |

### Rang 1 — taille du groupe de parents

**Mécanisme.** NEAT trie les membres de chaque espèce par fitness, copie jusqu’à deux élites, puis ne tire les parents que parmi les `ceil(survival_threshold × taille)` premiers, avec un minimum de deux. Au checkpoint G12, le seuil 0,30 donne 98 parents admissibles cumulés (18 pools de 2, 18 de 3, 2 de 4) ; 0,45 en donne 136 (3 de 2, 15 de 3, 15 de 4, 5 de 5) et change le pool de 35 espèces. Cela ne change ni les quotas d’espèces ni les 76 copies élites/180 nouveaux enfants de la reproduction de référence.

**Hypothèse testable.** B augmente la variété de parents et de topologies dans les enfants sans diluer leur score d’ancrage moyen ; la fitness et la survie appariées de validation restent au moins au niveau du contrôle. Un groupe de parents plus large peut aussi conserver des génotypes moins adaptés et affaiblir la pression de sélection.

**A/B borné.** Deux bras depuis le même checkpoint immuable et la même graine d’entraînement, un seul changement 0,30→0,45, une génération chacun. Enregistrer les parents distincts réellement tirés, leurs paires, les topologies, la fitness d’ancrage moyenne et la survie. Comparer les champions sur la même validation de 32 cartes à 938271 avec différence appariée. Arrêter après une génération ; ne pas conclure à un avantage si l’intervalle apparié inclut zéro ou si le score moyen d’ancrage recule.

### Rang 2 — ajout de neurones

**Mécanisme.** Avec `single_structural_mutation=False`, NEAT tire indépendamment le test Bernoulli `node_add_prob` pour chaque mutation d’enfant ; une addition de nœud désactive un lien et en crée deux. `conn_add_prob=0,50` reste inchangé et peut s’ajouter à cette mutation. Le comportement hérité peut évoluer plus vite en complexité, au prix de mutations perturbatrices et d’un peu plus de travail réseau.

**Hypothèse testable.** B augmente les nœuds cachés et les topologies actives, puis améliore le score d’ancrage et la validation par rapport au contrôle. Si seule la complexité augmente, ou si l’amélioration apparaît uniquement sur le score d’entraînement, l’essai n’est pas concluant.

**A/B borné.** Même départ immuable, mêmes ancres/seed, un seul changement 0,15→0,25, une génération par bras, puis les 32 cartes à 938271. Comparer moyenne et médiane des nœuds cachés, topologies distinctes, fitness d’ancrage, survie, morts et fitness appariée de validation. Ne pas augmenter à nouveau `node_add_prob` si une génération ne donne pas de gain apparié clair.

### Rang 3 — plancher de descendants par espèce

**Mécanisme.** Dans la reproduction NEAT-Python 1.1.0 installée, le plancher effectif est `max(min_species_size, elitism)`. Les allocations d’espèces sont normalisées pour atteindre 256 ; `WindowedStagnation.species_elitism=4` est une protection contre la stagnation distincte et resterait inchangée. La contre-factuelle G12 ci-dessus établit que le plancher ne s’appliquait pas aux espèces observées.

**Hypothèse testable.** D’autres distributions de fitness pourraient rendre le plancher actif ; dans ce cas, 4→2 peut libérer des places au profit des espèces performantes. Le snapshot G12 prédit toutefois peu d’effet pour cette population.

**A/B borné.** Ne lancer que si un futur checkpoint montre des espèces dont l’allocation théorique touche réellement le plancher. Depuis un même checkpoint, comparer 4 et 2 pendant une génération, suivre les allocations et nouveaux enfants, puis utiliser les mêmes 32 cartes à 938271. Si aucune allocation n’est au plancher, fermer l’essai sans validation coûteuse.

## Budget, compatibilité et décision

Les cinq dernières durées terminées des lignes G1–G12 ont une médiane d’environ 1 006 s par génération (~16,8 min) pour 256 génomes et cinq épisodes. Un contrôle commun plus les trois variantes représentent au moins quatre générations complètes, environ 67 minutes de simulation au rythme mesuré, auxquelles s’ajoutent validation, checkpoints et éventuel ralentissement. Le run CUDA principal était encore actif à la lecture ; lancer ces bras sur le même GPU ferait concurrence à la course. **Aucun essai d’entraînement supplémentaire n’est donc recommandé tant que la course principale utilise ce GPU ou qu’il ne reste pas explicitement une fenêtre après elle.** Si une fenêtre apparaît, prioriser 0,30→0,45 puis 0,15→0,25 ; ne tester 4→2 que si un futur audit constate un plancher actif. Pour chaque comparaison, utiliser la graine 938271 ; ne pas utiliser 741852963.

La commande actuelle `--initialize-from` préserve la configuration NEAT embarquée au checkpoint et n’offre pas de surcharge de paramètres NEAT ou de protocole. Pour effectuer ces A/B de façon reproductible, il faut un outil expérimental explicite qui inscrit le diff de configuration et son hash dans le nouveau run. Ne pas modifier le checkpoint, les settings ou le protocole du run source. Pour un changement de récompense, créer en plus une version de récompense/protocole distincte et rescorer tous les modèles ; la reprise stricte et le warm-start actuels refusent le mélange des objectifs.

La valeur `compatibility_threshold` n’est pas parmi les trois essais immédiats : la cible de 8–12 espèces est déjà dépassée, mais il n’y a pas de population d’espèces minuscules qui montre une perte de niches. `AdaptiveSpeciesSet` conserve un représentant de chaque espèce existante avant d’assigner les autres génomes ; augmenter le seuil ne fait donc pas disparaître à lui seul les espèces déjà présentes. Une branche de seuil serait moins interprétable qu’un test du plancher de reproduction.

## Références

- Run et champs source : [`history.jsonl`](../../runs/20260924-163112-852097/history.jsonl), [`settings.json`](../../runs/20260924-163112-852097/settings.json), [`episodes/`](../../runs/20260924-163112-852097/episodes/), [audit G10](g10-audit.md) et [carte des paramètres](parameter-map.md).
- Code local : [`evolution.py`](../../slitherai/evolution.py), [`train.py`](../../slitherai/train.py), [`evaluation.py`](../../slitherai/evaluation.py), [`neat.ini`](../../configs/neat.ini) et [`warmstart.py`](../../slitherai/warmstart.py).
- NEAT-Python officiel version 1.1.0 : [reproduction et seuil de parents](https://github.com/CodeReclaimers/neat-python/blob/v1.1.0/neat/reproduction.py), [mutations structurelles](https://github.com/CodeReclaimers/neat-python/blob/v1.1.0/neat/genome.py), [sémantique des paramètres de reproduction](https://github.com/CodeReclaimers/neat-python/blob/v1.1.0/docs/config_file.rst).
