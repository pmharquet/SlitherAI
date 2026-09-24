# Export-v1 pilot: G1–G5 paired audit

## Résultat

Les validations utilisent les mêmes 32 cartes, graine `938271`, sur 90 s : pilote G1 (`generation-0000.json`), pilote G5 (`generation-0004.json`) et référence legacy G15 (`generation-0014.json`). Les deltas ci-dessous sont appariés carte par carte, avec `n=32`, erreur-type de la moyenne des différences et intervalle t bilatéral à 95 % (`df=31`, `t=2,0395`). Les proportions de survie et de mort sont en points de pourcentage (pp). Tous les intervalles couvrent zéro.

| Mesure, delta B−A | Pilote G5 − pilote G1 | Pilote G5 − legacy G15 |
|---|---:|---:|
| Fitness | +0,623 (SE 4,305 ; IC [−8,157 ; +9,403]) | −0,140 (SE 3,235 ; IC [−6,737 ; +6,456]) |
| Survie | 0,0 pp (SE 7,8 ; IC [−15,9 ; +15,9]) | −3,1 pp (SE 8,4 ; IC [−20,2 ; +14,0]) |
| Nourriture gagnée | +4,599 (SE 5,699 ; IC [−7,026 ; +16,223]) | +2,243 (SE 4,507 ; IC [−6,950 ; +11,436]) |
| Au moins une mort | 0,0 pp (SE 7,8 ; IC [−15,9 ; +15,9]) | +3,1 pp (SE 8,4 ; IC [−14,0 ; +20,2]) |
| Mort par bord | −6,3 pp (SE 6,3 ; IC [−19,0 ; +6,5]) | 0,0 pp (SE 6,4 ; IC [−13,0 ; +13,0]) |
| Mort par collision | +6,3 pp (SE 10,0 ; IC [−14,1 ; +26,6]) | +3,1 pp (SE 7,1 ; IC [−11,3 ; +17,6]) |

Le fitness moyen de validation progresse de 17,935 à 18,558 entre G1 et G5, mais avec une forte incertitude appariée. Face à legacy G15 (18,698), le pilote G5 est pratiquement à égalité sur cet échantillon. Il survit sur 14/32 cartes contre 15/32 pour legacy G15; les décès par bord sont identiques (12/32), et les collisions sont 6/32 contre 5/32. Ces résultats ne démontrent ni gain ni perte. L’amélioration du score moyen de sélection pendant le pilote n’a pas produit un gain de validation établi.

## Population et coût observés

| Point | Score de sélection moyen / meilleur | Validation : fitness, survie, nourriture | Espèces | Nœuds moyens | Connexions activées moyennes | Temps de génération |
|---|---:|---:|---:|---:|---:|---:|
| Pilote G1 (`history` génération 0) | −1,376 / 29,241 | 17,935 ; 43,75 % ; 28,767 | 10 | 2,906 | 105,191 | 1 384,66 s |
| Pilote G5 (génération 4) | 7,097 / 38,343 | 18,558 ; 43,75 % ; 33,366 | 12 | 4,023 | 101,727 | 1 503,40 s |
| Legacy G15 (génération 14) | 4,545 / 36,062 | 18,698 ; 46,875 % ; 31,123 | 38 | 3,148 | 104,770 | 1 044,89 s |

Les générations pilote G1 et G5 sont celles du même nouveau run, pas les générations historiques de la population source. Les nœuds moyens augmentent pendant le pilote, tandis que les connexions activées moyennes baissent légèrement. Les temps par génération sont descriptifs; les deux runs n’ont pas le même nombre de cartes de simulation d’entraînement (32 pour le pilote, 64 pour legacy), donc ce n’est pas une comparaison de débit contrôlée. Le nombre d’espèces est également descriptif et ne justifie pas à lui seul un réglage de seuil.

## Provenance et état

- Run pilote : [`20260924-export-v1-pilot-20260924-210008`](../../runs/20260924-export-v1-pilot-20260924-210008/). Il a été initialisé depuis le checkpoint 10 de [`20260924-163112-852097`](../../runs/20260924-163112-852097/) (SHA-256 `303e21c10f6b4a1900278f9d44d93204b584ba6ba334af03f5f6738f1967767e`). `initialization.json` confirme le passage `legacy-v1` → `export-v1`, la réinitialisation des statistiques de fitness/espèces et la conservation des génomes et du RNG.
- Le schéma pilote est `slither-neat-530-export-v1`; celui de la référence est `slither-neat-530-v1` (`legacy-v1`). Les deux gardent le protocole `common-reference-v2`. Le pilote s’entraîne sur 32 cartes, population 256, jeux de 90 s; la référence sur 64 cartes, population 256, jeux de 90 s. L’écart de schéma et de taille de lot empêche d’attribuer causalement les différences au seul capteur.
- Snapshots liés aux points comparés : pilote checkpoint 1 SHA-256 `e9356ab207fcb6740e73813a46e873552ed1ea62d4b7cde9a77d170b9953fe4b`; pilote checkpoint 5 `1b22c5ab8c4f4651ab4fc0f929d26267fc075a40fbaa55192eeaf044d3c369b4`; référence checkpoint 15 `dec703966f7dcef0995d510b6640e0ed5f53bcbe59fb01c6c072e495f431aed9`.
- État lu le 24 septembre à 23:52 Europe/Paris : pilote `completed`, génération d’index 4, cible 5, sans erreur; les processus pilote connus sont terminés. La référence reste `training`, génération 18/50, sans erreur; le worker d’entraînement PID 6820 est présent.
- `settings.json` du pilote porte `generations: 1`, alors que `status.json` indique la cible 5 et `history.jsonl` contient les générations 0 à 4 avec les checkpoints 1 à 5. Le fichier de paramètres reflète donc la dernière invocation d’une génération reprise depuis checkpoint 4, pas le budget cumulé; le budget initial n’est pas conservé dans ce fichier. L’association validation↔génération est confirmée par le reporter d’entraînement (`slitherai/train.py`, `TrainingReporter.post_evaluate`); les JSON de validation ne consignent pas d’identifiant de génome.

## Reproduction du calcul

Lire `per_map.fitness`, `per_map.alive`, `per_map.food_gain`, `per_map.border_death` et `per_map.collision_death` dans les trois JSON de validation liés ci-dessus. Pour chaque mesure, calculer les 32 différences B−A dans l’ordre des cartes; rapporter leur moyenne, `sample_sd(diff)/sqrt(32)` et `mean(diff) ± 2,0395 × SE`. « Au moins une mort » vaut 1 si bord ou collision vaut 1. Il s’agit d’intervalles t appariés approximatifs; aucune correction de multiplicité n’est appliquée.
