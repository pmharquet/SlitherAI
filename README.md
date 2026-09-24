# SlitherAI Lidar

## Laboratoire NEAT local

Le projet comprend une simulation locale et un entraînement NEAT sur CUDA : 64 arènes de 16 vers, 530 entrées, deux sorties (boost et direction de 0 à 1), réseaux récurrents évolutifs, visualisation et sauvegarde/reprise. Chaque arène évalue un candidat face à 15 adversaires fixes ; les 256 réseaux jouent chacun cinq parties comparables par génération. L'objectif récompense la croissance nette, et les espèces sont suivies avec un seuil adaptatif et une stagnation mesurée sur plusieurs générations. Lancer `./start.ps1`, puis ouvrir <http://127.0.0.1:8765>. L'entraînement démarre uniquement avec le bouton de l'interface. [Installation, fonctionnement et limites du simulateur](docs/training.md).

## Collecteur de parties réelles

Extension Chrome Manifest V3 de lecture du jeu pour enregistrer les observations et les actions **humaines**. Elle calcule 87 faisceaux autour du ver à une cadence adaptative selon le temps de calcul : 45 dans les ±45° devant (pas de 2°), 22 sur les côtés jusqu'à ±90° (pas de 4°) et 20 vers l'arrière jusqu'à ±170° (pas de 8°). Le cône de 20° centré pile derrière n'a aucun faisceau. Chaque faisceau conserve autant de retours ordonnés que nécessaire : boules, proies mobiles, têtes, portions de corps et bordure. Les objets visibles sont également listés avec leurs attributs, y compris les petites boules qui tombent entre les faisceaux.

L'extension lit l'état du client, dessine une couche transparente et stocke localement les mesures après un clic sur « Enregistrer les vecteurs ». Elle ne pilote pas le ver, ne crée pas d'image et ne transmet pas de données. Chaque session peut être exportée manuellement en JSONL.

## Données disponibles

- Chaque retour : classe, distance d'entrée et de sortie, taille, index d'objet et type de mesure (`intersection` géométrique ou `sector` pour une boule affectée à un faisceau angulaire).
- Boules et proies visibles : position, distance, taille `sz`, couleur et, si présents, angle et vitesse des proies.
- Vers visibles : identifiant local, tête et points du corps, direction, angle voulu, vitesse brute, échelle, rayon estimé, nombre de segments, fraction du dernier segment, score calculé si les tables du client existent.
- Joueur : les mêmes propriétés, plus rang, meilleur rang et nombre de joueurs du serveur lorsqu'ils sont exposés, ainsi que l'action humaine observée (pointeur et boost). Les dix scores numériques du classement, le zoom, le viewport, les FPS et quelques indicateurs de latence sont conservés lorsqu'ils existent.

Les rayons s'arrêtent au bord du **viewport**. `censored` indique qu'il ne faut pas supposer l'espace libre au-delà. Le rayon du corps et la limite de l'arène sont des estimations. [Schéma et calculs](docs/lidar.md).

Le lidar reste centré en coordonnées du jeu quelle que soit la forme de la fenêtre. L'overlay se recale sur la position et l'échelle réelles du canvas à chaque mesure. La caméra enregistre aussi une zone carrée inscrite et centrée sur la tête pour une future entrée de modèle uniforme, tout en conservant les observations des côtés d'un écran large.

## Utilisation

1. Charger le dossier [`extension`](extension) comme extension non empaquetée dans `chrome://extensions`, puis recharger la page du jeu.
2. Démarrer une partie. L'overlay montre les rayons et un aperçu de quatre retours au maximum par rayon ; le popup affiche le nombre d'objets, le nombre total de retours, la durée du calcul, la cadence, le score et le rang si disponibles.
3. Démarrer et arrêter l'enregistrement local depuis le popup.
4. Dans « Sessions locales », cliquer sur « Voir / exporter », puis sur « Télécharger le JSONL complet ». Le fichier contient une ligne `session`, puis une ligne `sample` par mesure. Les rayons binaires sont décodés en tableaux JSON lisibles. Envoyer ce fichier pour analyser la partie.

L'enregistrement se ferme automatiquement quand la partie se termine ; la session conserve alors son score final observé et son rang final connu.

Le stockage est IndexedDB dans l'extension. Le format `SLR2` est binaire et de longueur variable ; les listes d'objets et leurs métadonnées restent dans l'échantillon. [Sources et limites](docs/recherche.md).

## Vérification

`node tests/lidar.test.js` vérifie les intersections, l'ordre des couches, la présence de toutes les boules visibles, les tailles, le score et le rang, la censure, le codage binaire, les angles qui passent par 0°/360° et la cadence adaptative. Ces tests utilisent un état simulé. Les variables du client et les approximations géométriques restent à vérifier sur la version jouée.

Les [conditions d'utilisation de slither.io](https://slither.io/tos/) consultées le 23 septembre 2026 restreignent l'emploi d'extensions ; une collecte auprès de joueurs exige de clarifier ce point avec l'exploitant.
