# Audit de la référence G15–G20

## Résumé

La validation G20 est une répétition exacte de G15 : mêmes 32 scores par carte, même génotype champion (ID `2867`, empreinte canonique `_genome_hash` `6e500a41c100865e361bdf00f1cc3003ab377179a31f7e2aef2fc961ed33fae0`). Il a 3 nœuds et 109 connexions, dont 98 activées. `best-validation.pkl/json` reste donc celui de G15, index de génération 14. Le champion de sélection est aussi l’ID 2867 en G15 et G17–G20; G16 est ID `3036`.

La population moyenne progresse en fitness de sélection, survie et nourriture entre G15 et G20, même si le champion de validation ne change pas. Le score individuel reste très sensible au jeu tournant. Les données ne justifient pas de changement NEAT, récompense ou simulateur ce soir. Le nombre d’espèces est bien au-dessus de sa cible, mais les niches sont équilibrées et le seuil adaptatif monte encore.

## Résultats génération par génération

G15–G20 correspondent aux index `14–19` dans [history.jsonl](../../runs/20260924-163112-852097/history.jsonl) et dans les fichiers [episodes](../../runs/20260924-163112-852097/episodes/). Les moyennes d’épisodes couvrent 256 génomes × 5 parties par génération.

| Génération | Champion ID (score sélection) | Fitness de sélection moyen | Fitness moyen par partie | Nourriture | Survie | Mort bord | Mort collision | Espèces / seuil |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G15 | 2867 (36,062) | 4,545 | 4,673 | 19,62 | 9,1 % | 67,0 % | 23,9 % | 38 / 2,981 |
| G16 | 3036 (36,210) | 4,081 | 4,196 | 18,93 | 9,5 % | 66,3 % | 24,3 % | 38 / 3,061 |
| G17 | 2867 (40,778) | 5,067 | 5,254 | 19,97 | 10,7 % | 65,0 % | 24,3 % | 38 / 3,141 |
| G18 | 2867 (38,947) | 5,559 | 5,680 | 19,49 | 15,5 % | 61,7 % | 22,7 % | 38 / 3,221 |
| G19 | 2867 (39,047) | 6,739 | 6,813 | 20,93 | 15,5 % | 60,7 % | 23,8 % | 38 / 3,301 |
| G20 | 2867 (38,681) | 7,426 | 7,569 | 21,49 | 18,3 % | 63,1 % | 18,6 % | 38 / 3,381 |

De G15 à G20, la moyenne de sélection monte de 2,881 points; la fitness par partie monte de 2,896, la nourriture de 1,87 et la survie de 9,1 pp. Les morts par collision baissent de 5,3 pp; les morts par bord baissent de 3,8 pp, avec un rebond en G20. L’écart-type moyen des scores entre les cinq parties d’un génome reste élevé, entre 9,37 et 10,37.

### Ancres et jeu tournant

Quatre parties d’ancrage fixes pèsent 80 % du score agrégé; une partie tournante pèse 20 %. Les moyennes brutes par partie ci-dessous portent sur toute la population. Les ancres ont les mêmes scénarios d’une génération à l’autre; la graine du cinquième scénario change.

| Métrique | Ancres G15 → G20 | Tournante G15 → G20 |
|---|---:|---:|
| Fitness moyenne | 5,62 → 8,06 | 0,87 → 5,59 |
| Nourriture | 20,45 → 22,81 | 16,28 → 16,19 |
| Survie | 10,9 % → 14,9 % | 2,0 % → 31,6 % |
| Mort bord | 67,2 % → 62,8 % | 66,0 % → 64,5 % |
| Mort collision | 21,9 % → 22,3 % | 32,0 % → 3,9 % |

Pour le même génome ID 2867, le score d’ancrage agrégé est `40,4285` en G15 et G20, avec les mêmes résultats sur les quatre ancres. Son score tournant passe de `18,598` à `31,693`; il meurt au bord sur la partie tournante G15 et survit à celle de G20. Cette variation suffit à faire varier son score de sélection de `36,062` à `38,681`. C’est une preuve de sensibilité aux scénarios, pas une mesure de bruit du simulateur sur une même graine.

## Validation et identité du champion

Les validations de [G15](../../runs/20260924-163112-852097/validation/generation-0014.json) et [G20](../../runs/20260924-163112-852097/validation/generation-0019.json) sont identiques, y compris tous les vecteurs `per_map`; leurs fichiers ont le même SHA-256 `a632532313e66de28bfda73b0f5fe83cc7fe099c0965a509b27caff799b76e6c`. Elles donnent fitness `18,69798` (SE entre cartes `3,01708`), survie `46,875 %`, nourriture `31,12296`, mort bord `37,5 %` et mort collision `15,625 %`. Le SE décrit la dispersion entre cartes pour ce génome; l’IC t à 95 % de sa moyenne est environ `[12,54 ; 24,85]`, et ne mesure pas une différence entre générations.

L’ID a été retrouvé en prenant l’argmax du score dans les épisodes G15 et G20, puis vérifié dans les populations des checkpoints 15 et 20 et dans `best-validation.pkl` (génération 14, ID 2867). L’empreinte ci-dessus est calculée par la fonction commune [`_genome_hash`](../../slitherai/evaluate_holdout.py), sur les champs des nœuds (id, biais, réponse, activation, agrégation) et des connexions (source, cible, poids, état activé, innovation). Les SHA-256 des checkpoints 15 et 20 sont respectivement `dec703966f7dcef0995d510b6640e0ed5f53bcbe59fb01c6c072e495f431aed9` et `f9c31929635aeabe39b8359fd0069f1c9e4cf2cd8dfe542c4dda328d949858f7`; les génotypes des deux checkpoints ont la même empreinte `_genome_hash`. Le reporter valide le meilleur génome de chaque génération (`TrainingReporter.post_evaluate` dans [train.py](../../slitherai/train.py)); les JSON de validation n’enregistrent pas l’ID directement. Le G20 à égalité n’a pas remplacé le meilleur stocké, car le reporter ne sauvegarde qu’en cas de score strictement supérieur.

Le SE `3,017` décrit la dispersion des 32 scores par carte pour ce génome. Comme G15 et G20 réévaluent le même génome sur les mêmes cartes et produisent les mêmes valeurs, cette répétition n’ajoute pas un nouvel échantillon indépendant. Elle ne prouve ni surapprentissage ni généralisation à d’autres cartes.

## Diversité, seuil et décision

| Mesure | G15 | G20 | Observation G15–G20 |
|---|---:|---:|---|
| Espèces | 38 | 38 | 38 aux six générations, au-dessus de la cible 8–12 |
| Seuil de compatibilité au checkpoint | 2,981 (cp15) | 3,381 (cp20) | Hausse régulière de 0,08 par génération; plafond 4,0 |
| Espèces effectives | 36,92 | 36,80 | Entre 36,67 et 37,14 |
| Taille des niches | 5–11 | 5–11 | Entre 3 et 11 sur la fenêtre; aucun singleton |
| Part de la plus grande niche | 4,3 % | 4,3 % | Pas de concentration sur une seule niche |
| Nœuds moyens / connexions activées | 3,148 / 104,77 | 3,484 / 103,80 | Hausse légère des nœuds, connexions activées en baisse |

La cible d’espèces n’est pas atteinte, mais la diversité observée reste répartie. Les tailles sont celles des populations évaluées (species-history, générations 14–19); le seuil est lu dans cp15–cp20, après la spéciation vers la population suivante. Forcer une chute à 8–12 espèces risquerait de supprimer des niches sans preuve que cela améliore la validation; laisser le seuil adaptatif évoluer est préférable à une intervention cosmétique. Le progrès de la moyenne de population, les résultats de jeu variables et seulement six générations ne suffisent pas pour conclure à un plateau global ou à un défaut du simulateur. Poursuivre la mesure sur des validations indépendantes est plus informatif qu’un réglage ce soir.

### Note sur `sensor_chunk=8`

Le benchmark de débit séparé est maintenant répliqué : la seconde mesure, graine `2123456` et ordre 8→4, donne `37 064,63 ms` contre `25 827,55 ms`, soit `1,435×`, avec résumés finaux identiques; voir [performance-review.md](performance-review.md). Le run courant est à `sensor_chunk=4`. Ce résultat soutient un gain de débit pour le même calcul, pas un gain de qualité. Un warm-start à chunk 8 depuis checkpoint 20/21 remet toutefois fitness et fitness d’ancrage à zéro, invalide les comportements mémorisés, reforme les espèces à la génération 0 et remet leur âge/stagnation à zéro (`warmstart.py`; [politique de reset](warm-start-reset-policy.md)). Il préserve les génomes et le RNG, mais brise la continuité des statistiques de sélection. Si le débit justifie un cutover au prochain checkpoint, garder le checkpoint source comme contrôle et comparer la validation après réévaluation; cette décision est distincte d’un changement de récompense ou de paramètres NEAT. Aucun lancement ni réglage n’a été fait dans cet audit.

## Limites des données

- Une seule trajectoire d’entraînement et six générations observées; aucune réplication de l’évolution pour estimer sa variance.
- Les quatre ancres restent fixes et la partie tournante change à chaque génération. Les métriques détaillées par partie sont enregistrées, mais G15 et G20 ne partagent pas la même graine tournante.
- La validation est un jeu fixe de 32 cartes et n’est pas utilisée par la sélection; son égalité ici vient d’un champion cloné, pas de deux politiques indépendantes.
- État lu après validation G20 : status `training`, génération 20/50, sans erreur.
