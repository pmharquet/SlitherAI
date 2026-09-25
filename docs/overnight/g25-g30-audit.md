# Validation G25 contre G30

## Comparaison appariée sur les 32 cartes

Les fichiers [validation G25](../../runs/20260924-163112-852097/validation/generation-0024.json) (SHA-256 `b48deb420a45e55cb3cdd9c0d230392ad0e143b82f85d955d9d19aad3963057b`) et [G30](../../runs/20260924-163112-852097/validation/generation-0029.json) (SHA-256 `7e9641ba10f2a70ef23797a953246c852e808783bbdffd7210e31dbcc2c50458`) ont le même seed `938271`, 32 cartes et 90 s par carte. Deltas `G30 − G25` calculés carte par carte; IC t bilatéraux approximatifs à 95 % (`df=31`, `t=2,0395`). Pour les indicateurs binaires, les IC sont approximatifs et les nombres de cartes discordantes sont donnés.

| Mesure | G25 | G30 | Delta (SE; IC 95 %) | Répartition par carte |
|---|---:|---:|---:|---|
| Fitness | 21,263 | 16,725 | −4,538 (3,485; [−11,645; +2,570]) | G30 meilleur 15 / G25 meilleur 17 |
| Nourriture | 31,997 | 26,846 | −5,152 (3,948; [−13,203; +2,899]) | 14 / 18 |
| Vivant à 90 s | 53,1 % (17/32) | 50,0 % (16/32) | −3,13 pp (8,38; [−20,22; +13,97]) | 3 mieux / 4 moins bien / 25 égales |
| Mort par bord | 31,3 % (10/32) | 15,6 % (5/32) | −15,63 pp (6,52; [−28,93; −2,32]) | 5 de moins / 0 de plus / 27 égales |
| Mort par collision | 15,6 % (5/32) | 34,4 % (11/32) | +18,75 pp (8,32; [+1,77; +35,73]) | 7 de plus / 1 de moins / 24 égales |
| Virage moyen par décision | 18,94° | 32,08° | +13,14° (3,35; [+6,32; +19,97]) | plus élevé sur 21 / plus bas sur 11 |

Le déplacement bord/collision est réel dans les décomptes, mais pas établi avec précision : tests exacts appariés de McNemar, `p=0,063` pour les cinq diminutions nettes de morts par bord et `p=0,070` pour les collisions (7 hausses, 1 baisse). La survie totale ne baisse que d’une carte. Parmi les changements de catégorie, 3 cartes passent de mort par bord à mort par collision et 2 de mort par bord à vivante; 4 passent de vivantes à mortes par collision et 1 de collision à vivante. Les sept nouvelles collisions s’accompagnent chacune d’une baisse du virage moyen sur leur carte, alors que les trois passages d’un décès à vivant s’accompagnent d’une hausse. La moyenne globale de virage monte donc fortement, mais ce relevé n’appuie pas une explication simple où ce virage supplémentaire causerait les collisions.

Les IC fitness et nourriture couvrent zéro; le nombre de cartes gagnées/perdues est presque équilibré. Ces IC sont nominaux et ne corrigent pas la sélection des checkpoints parmi plusieurs validations observées.

## Génomes et évolution G25–G30

Le gagnant de sélection G25 est le génome `4107`; celui de G30 est `5545`. Ils sont tous deux présents respectivement dans checkpoint 25 (SHA-256 `b24ff774359493c47261014a90a617e2473844cb02e95c053b5f7813b84be47c`) et checkpoint 30 (`6318dc74fcde84421f782aa7e2d40c7b791fb56616354bcad2afd3aec4fceae`). Le meilleur fichier de validation reste G25 (génération index 24).

| Champion | Espèce / taille | Nœuds | Connexions totales / actives | Empreinte canonique du génotype |
|---|---:|---:|---:|---|
| G25, ID 4107 | 31 / 10 | 4 | 115 / 95 | `1101711c875b87308d7db7ec397316957aeae3b2f84c29215809e6eb212ce7a9` |
| G30, ID 5545 | 10 / 10 | 4 | 112 / 103 | `6b22a74f0c68201d4e5ca834938288ec0b2b087102e6d32693face576b30dd5c` |

Ces empreintes utilisent la fonction commune [`_genome_hash`](../../slitherai/evaluate_holdout.py), sur l’id/biais/réponse/activation/agrégation des nœuds et la source/cible/poids/état activé/innovation des connexions. Le génome G25/ID `4107` du checkpoint 25 est identique champ par champ à celui de `best-validation.pkl` (validation index 24); l’export `best-validation-network.json` correspond aussi exactement à ses poids et états activés. Le hash G25 cohérent dans les deux audits est donc `110171…212ce7a9`; les valeurs précédentes divergentes provenaient de sérialisations différentes, non d’un changement réel du génome. Les deux champions gardent les mêmes sorties `boost_probability` et `absolute_direction_turns`, mais ne sont pas le même réseau : 7 connexions partagent leurs extrémités et seulement 2 sont activées dans les deux. Les sorties enregistrées montrent aussi G30 sans boost dépensé sur les 32 cartes, contre `0,066` en moyenne pour G25. C’est compatible avec une modification de comportement émergent; cela ne prouve pas que la topologie ou le virage explique à elle seule la baisse de score.

| Génération | Espèces | Score moyen / ancrage moyen | Rotative moyenne (écart-type population) | Champion sélection : ID, score / ancrage / rotative |
|---|---:|---:|---:|---:|
| G25 | 38 | 10,06 / 10,48 | 8,39 (13,16) | 4107, 36,99 / 38,33 / 31,63 |
| G26 | 38 | 10,07 / 10,72 | 7,47 (17,66) | 2867, 38,95 / 40,43 / 33,04 |
| G27 | 38 | 11,21 / 11,54 | 9,87 (13,77) | 2867, 37,31 / 40,53 / 24,45 |
| G28 | 38 | 9,73 / 11,02 | 4,58 (14,79) | 2867, 36,49 / 40,43 / 20,73 |
| G29 | 38 | 9,32 / 11,42 | 0,92 (14,58) | 4814, 38,84 / 35,84 / 50,85 |
| G30 | 38 | 9,56 / 11,01 | 3,77 (14,16) | 5545, 37,13 / 39,91 / 26,03 |

Les quatre jeux d’ancrage gardent leurs seeds/focaux; le cinquième jeu change de seed et de focal chaque génération. Le score de sélection est `0,8 × ancrage + 0,2 × jeu rotatif` ([`evaluation.py`](../../slitherai/evaluation.py)). Ainsi, le score d’ancrage du génome 4107 reste exactement `38,333` lorsqu’il est réévalué de G25 à G29, tandis que son résultat rotatif varie de `−7,63` à `31,63`; son score de sélection varie alors de `29,14` à `36,99`. Sur les deux champions comparés, l’ancrage G30 monte de `1,57`, le jeu rotatif baisse de `5,59` et le score de sélection ne monte que de `0,14`. Ce jeu unique crée une forte variation de classement; ce n’est pas une répétition du même scénario.

La moyenne de population ne s’effondre pas sur ces six générations : le score moyen va de `9,32` à `11,21`, l’effectif reste à 38 espèces et la complexité moyenne reste près de `3,6–3,7` nœuds et `103–104` connexions. L’écart G25/G30 relève donc d’un changement de gagnant et de réponse aux cartes, pas d’une chute uniforme de la population. Il est compatible avec un compromis comportemental et une sensibilité aux scénarios, sans isoler la cause ni montrer une régression durable à partir d’un seul checkpoint.

## Décision

Conserver les paramètres et ne pas coder de logique d’action en réponse à cette comparaison. Le holdout final indépendant reste la comparaison décisive; les validations répétées et les scénarios rotatifs servent à suivre la trajectoire, pas à transformer ce seul écart en réglage.
