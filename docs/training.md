# Simulation locale et entraînement NEAT

Depuis PowerShell, dans `C:\Docker\SlitherAI`, lancer `./start.ps1`, puis ouvrir <http://127.0.0.1:8765>. La page permet de démarrer, mettre en pause, arrêter et reprendre une session. Son ouverture ne démarre aucun apprentissage. `./install.ps1` prépare un environnement Python et PyTorch CUDA sur une nouvelle machine.

Le panneau **Réseau du ver observé**, sous la simulation, montre le génome et les activations réellement évaluées du candidat de l'arène affichée. Le choix automatique suit ce réseau ; les adversaires gris utilisent une stratégie fixe et n'ont pas de réseau NEAT à inspecter. Les entrées peuvent être regroupées en six familles de capteurs et huit variables, détaillées pour les entrées reliées, ou affichées toutes ensemble. Les poids positifs sont verts, les négatifs roses, les connexions récurrentes sont en pointillés. Leurs sources utilisent la valeur du pas précédent. La luminosité suit l'activité, l'épaisseur le poids. Survoler un neurone affiche sa valeur, son biais et son état. Les coordonnées graphiques organisent la lecture : NEAT ne donne pas de position physique aux neurones. Le panneau est actualisé environ une fois par seconde, peut être agrandi, déplacé et zoomé. Pendant la validation ou après l'arrêt, il indique qu'il présente un instantané.

## Configuration de départ

- 64 arènes indépendantes, 16 vers chacune : 1 024 vers simultanés sur GPU, dont 64 candidats NEAT et 960 adversaires à stratégie fixe.
- Population de 256 génomes : chaque génome joue cinq parties privées par génération, soit 1 280 épisodes, répartis en lots de 64.
- Partie de 90 secondes simulées, décision toutes les 100 ms et trois sous-étapes physiques par décision.
- 50 générations par lancement ; quantité modifiable dans l'interface.
- Reproduction, mutations, spéciation et suivi des innovations : NEAT-Python 1.1.0 sur CPU. Simulation, capteurs et exécution groupée des réseaux : PyTorch sur CUDA. Repli CPU possible.

Le GPU détecté pendant la préparation est une NVIDIA RTX 4060 Laptop de 8 Go, avec PyTorch 2.8.0+cu129. Le pilote fournit la compatibilité CUDA ; aucun toolkit de compilation supplémentaire n'est nécessaire pour ce code.

## Entrées et actions

Contrat de **530 entrées** défini dans `slitherai/schema.py` et sauvegardé avec chaque session. Deux identités précisent la sémantique du corps propre : `slither-neat-530-v1` (`legacy-v1`) est le défaut historique du simulateur ; `slither-neat-530-export-v1` (`export-v1`) reproduit la géométrie propre observée dans l'extension 0.6.0. Le compte, l'ordre des entrées, les rayons, les variables globales et les actions restent les mêmes.

- 87 rayons : 45 dans la zone ±45° au pas de 2°, 22 jusqu'à ±90° au pas de 4°, 20 jusqu'à ±170° au pas de 8°. Cône mort derrière.
- Six valeurs par rayon : portée visible, proximité tête ennemie, corps ennemi, corps propre, bordure et intérêt de nourriture. Les proies sont regroupées avec la nourriture.
- Huit valeurs globales : vitesse brute normalisée, rayon du corps, longueur en segments, sinus/cosinus du cap absolu, sinus/cosinus de l'écart entre cap demandé et actuel, boost de l'action précédente.
- Total : **87 × 6 + 8 = 530**. Les deux valeurs ajoutées à la proposition de 528 entrées indiquent l'orientation absolue, indispensable pour relier des rayons orientés selon le ver à une sortie de direction absolue.

En `legacy-v1`, le corps propre utilise le rayon physique et le filtre historique basé sur la longueur le long du corps (`3 × rayon`). En `export-v1`, le simulateur utilise les points à au moins `2 × rayon` spatialement de la tête, des capsules de rayon `2 × rayon + 3`, et ne relie pas deux points séparés de 600 unités ou plus. L'absence d'un objet donne 0 ; sa proximité vaut `1 / (1 + distance / 100)`. L'intérêt d'une boule vaut `min(taille / 20, 1) × proximité`, en conservant le maximum du rayon. La portée est normalisée par `portée / (portée + 1000)` ; elle évite de confondre un écran court avec un espace observé très loin. Les rayons de nourriture utilisent les mêmes projections angulaires que l'extension. Les distances et rayons physiques restent des estimations. `export-v1` est opt-in et sa parité synthétique JavaScript/Python ne calibre pas encore rayon, points de corps ou vue dans une vraie partie.

Les deux sorties sont `[boost, direction]` : boost activé à partir de 0,5 ; direction de 0 à 1 sur un tour complet, avec **0 = droite, 0,25 = bas, 0,5 = gauche, 0,75 = haut**. 1 et 0 indiquent la même direction.

Les réseaux commencent sans neurone caché avec environ 10 % des connexions directes possibles. Les mutations ajoutent/suppriment connexions et neurones et modifient les poids. Les connexions récurrentes sont autorisées : le réseau peut développer une mémoire entre décisions. Les activations sont sigmoïdes et l'agrégation est une somme. Il n'y a pas de nombre fixe de couches cachées ni de plafond logiciel sur leur taille ; la mémoire disponible reste une limite pratique. La sortie CUDA est testée contre le réseau récurrent de référence NEAT-Python, y compris après ajout de neurones.

## Simulation et fidélité

La simulation inclut mouvement continu, vitesse de rotation limitée, corps suivant une trajectoire, croissance, ralentissement du virage avec la taille, boost consommant de la masse et déposant des boules, collisions avec les autres vers et la bordure, nourriture issue des morts, nourriture dispersée et concentrée par endroits, et petites proies mobiles. Le corps propre n'est pas mortel. Le lidar conserve néanmoins cette information géométrique.

Les corps sont des polylignes rééchantillonnées à distance régulière. Les collisions utilisent leurs capsules ; tous les segments visibles retenus par le rectangle de vue contribuent au lidar, sans plafond de voisins. Le nombre de points physiques est fixé à 96 par ver ; l'espacement augmente pour les très grands vers. Le canal de corps propre suit le `sensor_version` enregistré au run : le défaut reste `legacy-v1`; `export-v1` n'est pas activé sur les sessions existantes. Les masses lâchées à la mort représentent 70 % de la masse du ver. Les dépôts de boost utilisent un réservoir circulaire fini ; les plus anciens finissent par être remplacés.

Il s'agit d'une **approximation locale**, pas d'une copie vérifiée du moteur propriétaire. Le rayon d'arène par défaut est réduit à 2 400 unités, avec ±20 % de variation, pour provoquer des rencontres avec 16 vers. Les exports observaient une estimation de rayon autour de 31 899 unités dans le jeu ; un rayon de cet ordre est configurable avec `--arena-radius 31899`, mais demande de revoir la densité de nourriture et de joueurs. La conversion de vitesse (`speedRaw × 20`), l'accélération immédiate, le rayon de collision, la masse, la croissance, le zoom et la rotation demandent encore un étalonnage. Les délais réseau et comportements du serveur officiel ne sont pas simulés.

Les calculs de collisions et de lidar omettent les segments de corps inactifs au-delà de la longueur du plus grand ver vivant. Cette optimisation conserve la géométrie ; un test compare ses résultats au calcul sur tous les segments.

## Objectif de l'évolution

La récompense **growth-v2** par partie est :

```text
25 × asinh((masse mangée − masse dépensée en boost) / 25)
+ 0,02 × secondes en vie
+ 2 × racine(éliminations)
− 12 si mort
```

La croissance nette domine l'objectif. La fonction `asinh` atténue les gains extrêmes ; reprendre uniquement sa propre nourriture de boost donne une croissance nette nulle. Survivre passivement 90 secondes rapporte 1,8 point. Une élimination suivie de sa propre mort vaut −10 points avant croissance et survie. Les poids définissent l'objectif sans prescrire une trajectoire ni copier les actions humaines.

Le protocole **common-reference-v2** évalue chaque génome seul face à 15 adversaires de collecte/évitement fixes. Quatre scénarios sont conservés entre générations ; un cinquième change à chaque génération. Pour un scénario donné, chaque génome reçoit les mêmes conditions initiales et la même suite de tirages aléatoires, indépendamment de sa place dans le lot GPU. Ses actions peuvent ensuite faire diverger la partie.

Le score fixe vaut `0,5 × moyenne + 0,5 × médiane` des quatre parties fixes. La sélection utilise `0,8 × score fixe + 0,2 × score du scénario renouvelé`. La stagnation des espèces utilise seulement les scores fixes. Une génération coûte davantage de calcul qu'avec l'ancien protocole ; la barre de progression indique les scénarios, les lots et les épisodes terminés.

Le protocole expérimental **mixed-reference-v5** garde la population de 256 génomes et oppose chaque génome dans une partie de 90 secondes sur l'une de **256 cartes**, avec 16 vers par carte : exactement un candidat NEAT et 15 contrôleurs heuristiques. Chaque génome joue une fois par génération. Le mélange des génomes sur les cartes dépend de la graine et de la génération ; son siège tourne d'un cran à chaque génération. Les 16 sièges reçoivent ainsi 16 candidats chacun. Le capteur utilise `sensor_chunk=8`. La sélection utilise le score de cette partie ; la stagnation compare les rangs médians au sein de la génération. Le capteur, les observations et la récompense `growth-v2` restent identiques.

La validation v5 garde les **32 cartes fixes pendant 90 secondes**, au premier génome évalué puis toutes les cinq générations (G1, G5, G10 pour un run de dix générations). Elle utilise les mêmes procédures de validation et de référence que les autres protocoles.

À la première génération puis toutes les cinq générations, le meilleur réseau de sélection est évalué sur **32 arènes fixes pendant 90 secondes**, avec des graines distinctes de l'entraînement. Deux références, collecte/évitement et déplacement en cercle, passent cette même épreuve. `best-validation.pkl` conserve le meilleur résultat moyen. Cette épreuve sert à sélectionner un modèle : elle n'est pas un test final indépendant et ne prouve pas le transfert au jeu original.

## Espèces et indicateurs

Les croisements, mutations, innovations et la reproduction utilisent **NEAT-Python 1.1.0**. Le projet ajoute `AdaptiveSpeciesSet` pour ajuster le seuil de compatibilité et `WindowedStagnation` pour mesurer les progrès des espèces sur les parties fixes.

- Une espèce regroupe des génomes proches selon leurs gènes et leurs poids. Le seuil initial est calibré sur la population pour viser **8 à 12 espèces** ; il baisse de 0,08 quand leur nombre est trop faible et augmente de 0,08 quand il est trop élevé, dans les limites 0,3–4. La plage cible n'est pas garantie à chaque génération. Un plafond lié à l'effectif minimal empêche une allocation impossible des descendants.
- Le score d'une espèce est la moyenne des scores fixes de ses membres. Toutes les cinq observations, on compare la médiane des cinq dernières à celle des cinq précédentes. Un gain supérieur à **0,25 point** remet son compteur de stagnation à zéro. Le premier groupe de cinq établit la référence.
- **30 générations** sans progrès peuvent rendre une espèce éliminable. Les **4 meilleures espèces** selon leur score fixe moyen actuel sont protégées, et au maximum **une espèce par génération** est supprimée pour stagnation.
- La reproduction utilise la **fitness moyenne ajustée**, normalisée sur les fitness des espèces retenues. Les deux meilleurs génomes de chaque espèce sont conservés ; les parents sont sélectionnés dans les meilleurs 30 %, avec au moins deux candidats.
- Le nombre d'espèces, leur effectif, leur part maximale et leur diversité effective décrivent la population actuelle. La diversité effective vaut `exp(-Σ p ln p)` : elle vaut 10 pour 10 espèces d'effectifs égaux, et baisse lorsqu'une espèce domine.
- Le tableau affiche les scores de la dernière génération entièrement évaluée : meilleure/moyenne/ajustée, score fixe moyen, nourriture et survie, neurones cachés et liens moyens, âge NEAT, stagnation, protection et élimination. « Effectif → suivant » compare l'effectif évalué à celui obtenu après reproduction et nouvelle spéciation.
- Le ver évalué porte la couleur de son espèce ; les adversaires fixes sont gris. L'inspecteur indique aussi l'espèce du réseau observé.
- Les indicateurs de comportement montrent nourriture, survie, morts contre le bord ou un autre corps, fraction de décisions avec boost et variation angulaire de la direction demandée. La décomposition du score distingue croissance, survie, éliminations et décès. La dispersion des scores entre épisodes reste affichée.

`species.json` conserve la population actuelle et la dernière évaluation. `species-history.json` conserve l'évolution des effectifs et les détails des générations mesurées. Pour une session commencée avant cette fonctionnalité, les effectifs anciens sont récupérés depuis les checkpoints ; les fitness détaillées manquantes ne sont pas inventées. Les indices des fichiers commencent à 0 ; l'interface les affiche à partir de 1.

Le cinquième scénario change entre générations : une baisse ponctuelle du score de sélection ne suffit donc pas à conclure à une régression. Le score fixe et la validation complètent cette mesure. Les espèces continuent à évoluer, naître et disparaître ; leur nombre seul ne mesure pas la qualité du jeu.

## Fichiers de sauvegarde

Chaque session est stockée sous `runs/<date-heure>/` :

- `checkpoint-N` : population, espèces, configuration, innovations et générateur aléatoire ; N est la prochaine génération à évaluer. Écriture temporaire puis remplacement pour résister à une coupure pendant la sauvegarde.
- `initialization.json` (runs initialisés depuis un checkpoint) : provenance du checkpoint source et règles de remise à zéro. La population et ses gènes sont importés, les fitness/ancrages/comportements sont effacés, et les espèces sont recréées en génération zéro. L'historique, les épisodes, les baselines et la validation du run source ne sont pas copiés.
- `champion.pkl` : meilleur réseau de la dernière génération évaluée ; `champion-network.json` expose sa topologie.
- `best-validation.pkl` : meilleur réseau sur l'épreuve fixe, avec son résultat dans `best-validation.json`.
- `episodes/generation-NNNN.json` : résultats individuels des parties de chaque génome, scénarios, scores fixes et scores de sélection.
- `validation/` et `baselines.json` : résultats détaillés par arène des validations et des stratégies de référence.
- `history.jsonl`, `settings.json`, `schema.json`, `status.json` et `console.log` : métriques, paramètres et diagnostic.

Après arrêt ou extinction du PC, ouvrir l'interface et cliquer sur **Reprendre**. Une génération incomplète est recommencée. Les paramètres de simulation sauvegardés sont réutilisés ; « Générations » indique le nombre de générations supplémentaires à exécuter. Les fichiers pickle sont des sauvegardes Python locales : le chargeur n'est prévu que pour les sauvegardes créées par ce projet.

`sensor_chunk` choisit la taille des blocs de raycasts (`4`, `8` ou `16`) sans changer le schéma d'observation. Un nouveau run utilise `4` par défaut. En reprise stricte, l'option omise hérite de `settings.json` ; une valeur explicitement différente est refusée. Au warm-start, l'omission hérite aussi du checkpoint source, tandis qu'une valeur fournie sélectionne le lot du nouveau run. Les settings de destination enregistrent la valeur effective. Les anciens settings sans ce champ sont interprétés comme `4`.

`--initialize-from checkpoint-N` démarre une nouvelle session depuis toute la population vivante d'un checkpoint local. Ce n'est pas une reprise : `--resume` et `--initialize-from` sont exclusifs. Le run de destination doit être vide, la population conserve son effectif, les paramètres physiques et la récompense doivent correspondre ; seuls le nombre de cartes, la taille de lot du capteur et le mode capteur peuvent différer. Sans `--sensor-version`, le nouveau run hérite du mode source. Un changement de mode doit être indiqué explicitement. Le compteur d'innovation global, l'indexeur de neurones et l'état aléatoire Python du checkpoint sont conservés ; le cache de déduplication des innovations de la génération est vidé, et la spéciation est recalibrée sur la population importée. Les scores précédents ne sont jamais reportés.

Les versions de récompense et de protocole sont enregistrées dans `settings.json`. Les anciennes sessions sont conservées mais ne peuvent pas être reprises avec l'objectif v2 : leurs scores et compteurs de stagnation ne sont pas comparables. Leurs configurations restent lisibles pour l'analyse avec la récompense historique `legacy-v1` et `configs/neat-legacy.ini`.

En ligne de commande :

```powershell
.\.venv\Scripts\python.exe -m slitherai.train --run runs/experience --maps 64 --worms 16 --population 256 --generations 50 --seconds 90 --device cuda
```

Pour un nouveau run explicitement aligné sur la géométrie propre exportée, en important une population précédente :

Pour lancer dix générations expérimentales depuis `checkpoint-44` avec mixed-reference-v5 :

```powershell
.\.venv\Scripts\python.exe -m slitherai.train --run runs/mixed-reference-v5-cp44-10gen --initialize-from runs/20260924-163112-852097/checkpoint-44 --opponent-mode mixed-reference --training-games 1 --maps 256 --worms 16 --population 256 --generations 10 --seconds 90 --seed 1 --validation-every 5 --sensor-chunk 8 --device cuda
```

```powershell
.\.venv\Scripts\python.exe -m slitherai.train --run runs/warmstart-export --initialize-from runs/experience/checkpoint-72 --maps 64 --worms 16 --population 256 --generations 50 --seconds 90 --device cuda --sensor-version export-v1 --sensor-chunk 8
```

## Exploiter les enregistrements humains

```powershell
.\.venv\Scripts\python.exe -m slitherai.dataset "C:\chemin\partie.jsonl" --out runs/partie.npz
```

Le convertisseur produit les 530 entrées et les deux sorties sous le contrat `export-v1`, ainsi qu'un rapport de vitesse observée et de rotation. Il accepte le lidar à 87 rayons de l'extension 0.6.0. Le contrat capteur décrit ces enregistrements ; le simulateur reste par défaut en `legacy-v1` jusqu'à sélection explicite. Les exemples humains servent ici au contrôle des entrées et à l'étalonnage ; NEAT apprend en jouant dans le simulateur. Le convertisseur n'entraîne aucun réseau.

## Vérifications

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --test tests/lidar.test.js tests/network-view.test.js
```

Tests : géométrie, conservation d'une prise alimentaire disputée, collisions, corps propre traversable, boost, orientation circulaire, cône mort, indépendance des cartes, entrées réelles, calcul récurrent CPU/CUDA, récompense de croissance, conditions communes indépendantes du lot CPU/CUDA, agrégation des scores, stagnation et protection, seuil adaptatif, reprise déterministe de l'évolution et absence de lancement lors de l'ouverture de l'interface.

Sources techniques : [configuration officielle NEAT-Python](https://neat-python.readthedocs.io/en/latest/config_file.html), [code officiel NEAT-Python](https://github.com/CodeReclaimers/neat-python), [installation PyTorch](https://pytorch.org/get-started/locally/). Les valeurs du simulateur sont des paramètres du projet, pas des constantes garanties par ces sources.
