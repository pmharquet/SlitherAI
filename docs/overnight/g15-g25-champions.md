# Comparaison des champions G15 et G25

## Validation appariée

Les candidats sont les gagnants de sélection des générations G15 (index 14, ID `2867`) et G25 (index 24, ID `4107`). Les fichiers [validation G15](../../runs/20260924-163112-852097/validation/generation-0014.json) (SHA-256 `a632532313e66de28bfda73b0f5fe83cc7fe099c0965a509b27caff799b76e6c`) et [validation G25](../../runs/20260924-163112-852097/validation/generation-0024.json) (SHA-256 `b48deb420a45e55cb3cdd9c0d230392ad0e143b82f85d955d9d19aad3963057b`) utilisent la même graine `938271`, les mêmes 32 cartes et 90 s par carte. Deltas `G25 − G15`, appariés sur chaque carte; IC t bilatéraux approximatifs à 95 % (`df=31`, `t=2,0395`). Les proportions sont en points de pourcentage (pp).

| Mesure | G15 | G25 | Delta (SE ; IC 95 %) | Cartes dans le sens G25 / G15 / égales |
|---|---:|---:|---:|---:|
| Fitness | 18,698 | 21,263 | +2,565 (3,526 ; [−4,626 ; +9,757]) | 21 / 11 / 0 |
| Nourriture gagnée | 31,123 | 31,997 | +0,874 (4,366 ; [−8,030 ; +9,778]) | 19 / 13 / 0 |
| Survie | 46,9 % | 53,1 % | +6,25 pp (8,91 ; [−11,92 ; +24,42]) | 5 / 3 / 24 |
| Mort par bord | 37,5 % | 31,25 % | −6,25 pp (8,91 ; [−24,42 ; +11,92]) | 5 de moins / 3 de plus / 24 égales |
| Mort par collision | 15,6 % | 15,6 % | 0,0 pp (7,78 ; [−15,86 ; +15,86]) | 3 de moins / 3 de plus / 26 égales |
| Toute mort | 53,1 % | 46,9 % | −6,25 pp (8,91 ; [−24,42 ; +11,92]) | 5 de moins / 3 de plus / 24 égales |

G25 est meilleur en fitness sur 21 des 32 cartes et survit sur deux cartes de plus, mais tous les IC incluent zéro. Les résultats indiquent un signal favorable, pas un gain généralisable établi.

## Génotype, espèce et sorties d’action

Les scores des fichiers [`episodes/generation-0014.json`](../../runs/20260924-163112-852097/episodes/generation-0014.json) et [`episodes/generation-0024.json`](../../runs/20260924-163112-852097/episodes/generation-0024.json) identifient ces deux champions. Le génotype G15 est retrouvé dans checkpoint 15 (SHA-256 `dec703966f7dcef0995d510b6640e0ed5f53bcbe59fb01c6c072e495f431aed9`); G25 et le payload `best-validation.pkl`/`best-validation-network.json` sont ID `4107`, validation génération 24, checkpoint 25 (SHA-256 `b24ff774359493c47261014a90a617e2473844cb02e95c053b5f7813b84be47c`). Les deux candidats sont donc bien ceux mesurés.

| | G15 / ID 2867 | G25 / ID 4107 |
|---|---:|---:|
| Empreinte canonique des gènes | `6e500a41c100865e361bdf00f1cc3003ab377179a31f7e2aef2fc961ed33fae0` | `1101711c875b87308d7db7ec397316957aeae3b2f84c29215809e6eb212ce7a9` |
| Nœuds (sorties et cachés) | 3 | 4 |
| Gènes de connexion / activés | 109 / 98 | 115 / 95 |
| Espèce / taille dans le checkpoint | 14 / 11 | 31 / 10 |
| Seuil du checkpoint | 2,981 | 3,781 |

Ces empreintes utilisent la fonction commune [`_genome_hash`](../../slitherai/evaluate_holdout.py), sur les champs de nœud (id, biais, réponse, activation, agrégation) et de connexion (source, cible, poids, état activé, innovation). La copie G25/ID `4107` du checkpoint 25 et le génome de `best-validation.pkl` sont identiques champ par champ; `best-validation-network.json` correspond également aux mêmes nœuds, connexions, poids et états activés. Le nœud caché G15 (`7`) disparaît; G25 ajoute les nœuds `353` et `500`. Les réseaux partagent 11 connexions par extrémités; 85 connexions activées sont propres à G25 et 88 propres à G15. Les 11 poids communs diffèrent (écart absolu moyen `0,192`). C’est une modification de politique/topologie, pas le même champion réévalué. Les IDs d’espèce sont des étiquettes; on ne peut pas interpréter à eux seuls un passage d’une niche à l’autre.

Les deux réseaux gardent les mêmes sorties `boost_probability` et `absolute_direction_turns`. Sur la validation, le `turn_degrees` moyen passe de `16,38°` à `18,94°` par décision (+`2,56°`, SE `1,58`, IC 95 % [−`0,67°`; +`5,79°`]); il est plus élevé sur 23 cartes et plus bas sur 9. Le boost reste rare : `boost_fraction` vaut `0,1476 %` puis `0,1463 %` des décisions; `boost_spent` moyen baisse de `0,1313` à `0,0656`, avec 9 cartes utilisant du boost pour G15 et 5 pour G25. Ces agrégats sont compatibles avec une politique qui tourne davantage et booste peu, mais aucun journal d’actions pas à pas n’est sauvegardé pour attribuer la hausse de fitness à une stratégie précise.

## Lecture et holdout final

Le changement de génome et les 21 cartes favorables rendent crédible une différence de comportement; les différences de survie, virages et morts restent toutefois incertaines sur 32 cartes. G25 est aussi le meilleur candidat de validations répétées à graine fixe (le fichier `best-validation.json` indique génération 24). L’IC ci-dessus ne corrige pas ce choix parmi plusieurs checkpoints et peut surestimer son gain.

Pour choisir un modèle final, comparer G15 et G25 une seule fois, de façon appariée, sur le jeu final indépendant qui n’a servi ni à la sélection ni aux validations périodiques. Garder G25 comme candidat prometteur, pas comme gagnant déjà confirmé. Ne pas retoucher les paramètres à partir de ces seuls résultats.

### Indicateurs appariés complémentaires

| Action | G15 | G25 | Delta (SE ; IC 95 %) | Répartition par carte |
|---|---:|---:|---:|---:|
| Boost dépensé moyen | 0,1313 | 0,0656 | −0,0656 (0,0481 ; [−0,1637 ; +0,0324]) | 4 plus / 8 moins / 20 égales |
| Fraction de décisions avec boost | 0,1476 % | 0,1463 % | −0,0014 pp (0,1013 ; [−0,2080 ; +0,2052]) | 4 plus / 9 moins / 19 égales |
| Degrés de virage moyen par décision | 16,382 | 18,941 | +2,559 (1,582 ; [−0,667 ; +5,785]) | 23 plus / 9 moins |

L’audit utilise uniquement les JSON, checkpoints et réseaux sauvegardés, sans simulation ni GPU. L’état du run lu après G25 est `training`, génération 26/50, sans erreur.
