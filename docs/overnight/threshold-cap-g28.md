# Seuil de compatibilité au plafond — checkpoint 28

## Résultat de reclassification CPU

Le checkpoint `runs/20260924-163112-852097/checkpoint-28` (SHA-256 `ab0ea16a3c6a0f0d371e23ed5e8c5b9e53b2dde0213897cb9c87bda24334dd5c`) contient la population de 256 génomes avant l’évaluation G28. Son seuil est déjà au plafond `4.0`; la cible déclarée est 8–12 espèces. J’ai chargé séparément quatre copies en mémoire. Pour neutraliser le retour adaptatif pendant ce test statique, j’ai fixé temporairement la cible min/max à 38 et le plafond au seuil essayé, puis appelé `AdaptiveSpeciesSet.speciate` sur les mêmes génomes. Cela laisse le code de classement en place tout en appliquant réellement chacun des seuils 4, 5, 6 et 8. Aucune simulation, évolution, mutation du run ou exécution GPU n’a eu lieu.

| Seuil testé | Espèces | Tailles (effectifs) | Effectif Shannon / Simpson | Part max. | ARI vs partition sauvegardée |
|---:|---:|---|---:|---:|---:|
| 4 | 38 | 5×11, 6×13, 7×2, 8×3, 9×5, 10×4 | 36,82 / 35,66 | 3,91 % | 1,00 |
| 5 | 38 | identiques | 36,82 / 35,66 | 3,91 % | 1,00 |
| 6 | 38 | identiques | 36,82 / 35,66 | 3,91 % | 1,00 |
| 8 | 38 | identiques | 36,82 / 35,66 | 3,91 % | 1,00 |

ARI compare les partitions en ignorant les étiquettes numériques d’espèce. Les deux effectifs sont l’exponentielle de Shannon et l’inverse de Simpson. Aucune niche n’a moins de cinq membres; la population est répartie presque uniformément. Dans la reclassification à seuil 4, la distance au représentant de sa niche est au plus `0,806` pour chaque membre. Les représentants des 38 niches existantes sont eux-mêmes tous à moins de 4 les uns des autres (703 paires, maximum `3,134`). Pourtant, l’algorithme garde un représentant pour chaque espèce précédente : ni le seuil 4 ni 8 ne fusionne ces niches. Sur cette population précise, augmenter le plafond ne réduit donc ni le nombre de niches ni le plancher de reproduction qui leur est associé.

## Ce qui peut faire baisser le compte

Dans [AdaptiveSpeciesSet](../../slitherai/evolution.py), le seuil règle l’affectation des génomes non encore représentés et la création de nouveaux fondateurs; chaque espèce existante reçoit d’abord un représentant. La cible 8–12 pilote le seuil par pas de `0,08`, mais ne garantit donc pas que le compte descende jusqu’à cette plage. La [configuration](../../configs/neat.ini) fixe aussi `max_stagnation=30`, protège les quatre espèces les mieux classées et limite les retraits à une espèce par génération. La remise à zéro complète ne s’applique qu’après extinction de toutes les espèces. Au G28, les espèces ont entre 1 et 14 générations depuis leur dernière amélioration; aucune n’atteint actuellement le seuil de stagnation de 30. L’historique montre 38 espèces sans création ni retrait pendant G20–G27. Ces observations décrivent des mécanismes distincts; elles ne permettent pas d’attribuer à la seule limite 4 le maintien du compte à 38.

## Allocation et décision

L’allocation exacte n’est pas reproductible depuis ce checkpoint : il ne contient une fitness que pour 76 des 256 génomes; les 180 nouveaux génomes n’ont pas encore été évalués à l’entrée de G28. Sans les scores, le fitness ajusté par niche et les tailles de descendance calculées par `DefaultReproduction` sont inconnus. Les contraintes mécaniques restent calculables : si les 38 niches sont conservées, `min_species_size=4` impose un plancher de 152 individus, et `elitism=2` conserve 76 élites; le pool de sélection compterait 2 ou 3 membres par niche (90 places au total) avec `survival_threshold=0,3`, mais leur identité et les allocations au-delà du plancher dépendent des scores.

Je ne recommande pas de relever le plafond maintenant. Cela ne change pas la partition G28; les niches existantes ne disparaissent pas par reclassement, tandis qu’un seuil plus haut rendrait plus difficile la création future de niches pour des innovations structurelles. C’est précisément le rôle protecteur de la spéciation décrit par [Stanley et Miikkulainen (2002)](https://doi.org/10.1162/106365602320169811) et la [FAQ officielle NEAT-Python 1.1.0](https://neat-python.readthedocs.io/en/v1.1.0/faq.html). L’effet d’un seuil plus haut sur les générations futures n’a pas été simulé; l’absence de nouveaux IDs sur G20–G27 rend en outre un bénéfice immédiat peu plausible. Garder le réglage actuel et réévaluer après davantage de données est la décision la moins risquée.

## Reproduction de l’audit

Depuis la racine du dépôt, avec l’environnement du projet (`.venv/Scripts/python.exe` sous Windows), charger indépendamment le tuple gzip/pickle du checkpoint pour chacun des seuils `4, 5, 6, 8`. Dans la copie, fixer le seuil testé, son plafond au même nombre, et `target_min=target_max=38` afin que le retour adaptatif ne remplace pas la valeur avant le classement; puis appeler `species.speciate(config, population, generation)`. Les calculs de diversité utilisent l’entropie de Shannon exponentiée et l’inverse de Simpson sur les tailles de niches; aucun état modifié n’est réécrit.
