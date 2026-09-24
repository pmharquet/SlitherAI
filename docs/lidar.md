# Contrat technique du lidar

## Lecture et fréquence

`lidar-main.js` s'exécute dans le monde JavaScript de la page et lit les valeurs déjà reçues par le client. Il ne les modifie pas. Il calcule un état transmis à `content.js`, qui dessine l'overlay et, sur demande, enregistre l'échantillon. Le prochain calcul démarre après une pause d'au moins 100 ms et d'environ huit fois le coût du calcul et de la transmission, plafonnée à 750 ms ; en dehors d'une partie ou après une erreur, la pause est de 500 ms. La cadence effective est donc variable et mesurée dans `performance.intervalMs`, avec le coût du calcul dans `performance.computeMs`. La source principale est `snake`/`slither`, `snakes`/`slithers`, `foods`, `preys`, `grd`, `gsc`, `view_xx`, `view_yy`, `rank`, `snake_count`, `fpsls` et `fmlts`. Une donnée manquante reste `null` ; elle n'est pas inventée.

L'action humaine enregistrée comprend la position du pointeur, son angle relatif au cap et le boost observé par bouton gauche ou Espace. Aucun événement de commande n'est synthétisé. Les pseudos ne sont pas stockés.

## Géométrie et couverture

- Origine : tête du joueur, `xx + fx`, `yy + fy` quand les corrections existent.
- Directions : 45 axes de −44° à +44° (pas de 2°, zone ±45°), 11 axes de chaque côté de 48° à 88° (pas de 4°, zone jusqu'à ±90°), puis 10 axes de chaque côté de 94° à 166° (pas de 8°, zone jusqu'à ±170°), relatifs à `ang`. Aucun axe dans le cône arrière de 170° à 190°.
- Têtes : disques. Corps : capsules entre points voisins connectés et disques aux points isolés, avec fusion des intersections qui se recouvrent pour un même ver. Deux portions séparées du même corps restent deux retours.
- Nourriture : chaque boule visible hors du cône mort est affectée au faisceau angulaire le plus proche. Les boules sont aussi conservées dans `visibleFood` : cette liste reste exhaustive pour le viewport, y compris dans le cône mort. Un retour `sector` mesure la profondeur projetée sur l'axe et ne prétend pas être une intersection exacte.
- Les angles sont ramenés sur un tour complet avec un modulo positif avant l'affectation aux faisceaux. Une direction qui passe de 360° à 0° conserve ses retours.
- Proies mobiles : même traitement que la nourriture, avec classe `prey` et propriétés cinématiques lorsqu'elles existent.
- Bordure : cercle de centre `(grd, grd)`, rayon estimé `0,98 × grd`.
- Portée : limitée à la sortie du viewport sur chaque axe. `censored: true` signifie que l'au-delà n'est pas observé. Il n'y a pas de plafond pratique inférieur à la taille de l'écran.

## Fenêtre, centrage et zone carrée

Les intersections restent en **unités du jeu**, centrées sur la tête : leurs angles et distances ne dépendent pas des dimensions CSS de la fenêtre. À chaque mesure, `cameraState` lit la taille interne de `mc` et son `getBoundingClientRect()`. La projection suit deux étapes : monde → coordonnées internes du canvas (`mww2/mhh2`, `view_xx/view_yy`, `gsc`), puis canvas → pixels visibles (`rect.left/top`, `rect.width/mc.width`, `rect.height/mc.height`). Les distances visibles sont coupées à l'intersection du canvas avec le viewport. L'overlay et l'angle du pointeur utilisent cette même transformation, y compris si la page redimensionne le canvas de façon différente sur chaque axe. Sans canvas utilisable, un repli sur `ww/hh` est signalé par `camera.layoutSource`.

`camera.squareHalfExtentWorld` est la demi-largeur du plus grand carré, centré sur la tête et entièrement visible dans le viewport courant. Pour préparer plus tard une entrée carrée uniforme, la portée de ce carré dans une direction `θ` vaut `squareHalfExtentWorld / max(|cos θ|, |sin θ|)` ; on peut conserver les retours avant cette portée et normaliser les distances par la demi-largeur. L'extension garde aussi **toute** la zone réellement visible, afin de ne pas perdre les objets des côtés d'un écran large. Le carré ne révèle pas une région qui serait hors écran ; sa taille est recalculée après chaque recadrage et chaque changement de zoom.

Pour la distance libre du **centre de la tête**, les obstacles dangereux sont agrandis du rayon estimé du joueur et d'une marge de trois unités. Le rayon visuel d'un ver est approximé par `14,5 × sc`. Ces nombres sont issus de mods anciens et doivent être étalonnés sur le client actuel. Les boules et proies gardent leur `sz` brut ; leur `radiusEstimate` est utilisé pour le dessin du faisceau.

Un rayon contient `relativeAngle`, `range`, `censored`, `distance` et `kind` pour le premier danger, les distances de première rencontre par classe pour compatibilité, puis `returns[]` avec autant de couches que rencontrées (maximum technique de 65 535 dans le format binaire). Chaque retour contient `kind`, `distance`, `exitDistance`, `objectIndex`, `size`, `sampling`. Les retours sont triés par entrée croissante. `objectIndex` relie une tête ou un corps à `visibleSnakes[]`, et une boule ou une proie à `visibleFood[]`. `-1` désigne la bordure ou le corps propre. La nourriture ne masque jamais un danger plus loin.

L'overlay limite son aperçu à quatre marqueurs par faisceau pour préserver la fluidité. `returns[]`, `visibleFood[]` et les mesures enregistrées conservent tous les retours.

## Attributs supplémentaires

`visibleFood[]` fournit `source`, position absolue et relative, distance, `sizeRaw`, `radiusEstimate`, code de couleur, puis angle et vitesse bruts si présents. `visibleSnakes[]` fournit identifiant de la session de jeu, tête, indicateur `headVisible`, points visibles du corps, cap, cap demandé, cap effectif, direction, vitesse brute, échelle, rayon estimé, `sct`, `fam`, code de couleur et score. Le joueur possède aussi `rank`, `snake_count` et `best_rank` si ces variables existent. `camera` conserve le zoom, le centre de vue, les dimensions et la position de la tête à l'écran. `telemetry` conserve les valeurs facultatives FPS, multiplicateur de latence, identifiant du serveur, souris du client et géométrie estimée de l'arène.

Le score porte `source: client_formula` seulement si `sct`, `fam`, `fpsls[sct]` et `fmlts[sct]` sont valides. Formule : `floor(15 × (fpsls[sct] + fam / fmlts[sct] − 1) − 5)`. Si ces tables manquent pour le joueur, la longueur numérique affichée dans `span_length` sert de repli (`displayed_length`) ; sinon `value: null, source: unavailable`. Le rang global n'est fiable que pour le joueur local (`rank`) ; les autres vers ont `rank: null`. Les scores locaux des vers visibles ne permettent pas de reconstituer le classement de tout le serveur. `snake_count` donne le dénominateur si disponible. Les dix scores numériques affichés dans `lbs` sont lus séparément quand l'élément existe, sans les noms et sans les rattacher aux vers visibles.

Certains attributs du client peuvent exister alors que leur objet n'est que partiellement visible. Ils restent distincts des mesures géométriques de l'écran ; l'apprentissage par imitation peut filtrer les champs qu'un humain ne verrait pas. Aucun temps avant collision mobile n'est annoncé comme exact.

## Format stocké

Après clic sur l'enregistrement, chaque échantillon contient l'horodatage, l'état du joueur, les listes `visibleSnakes` et `visibleFood`, les scores du classement si lisibles, la géométrie, les compteurs, un bloc `rayBytes` et l'action humaine. Une session se ferme automatiquement lorsque le client signale la fin de la partie ; elle retient le dernier score et le dernier rang observés, ainsi que la raison de fermeture. La page d'export lit cette session dans IndexedDB et produit un fichier JSONL : une ligne `session`, puis une ligne `sample` par mesure, avec les `rayBytes` décodés en `lidar.rays[]`. L'aperçu est tronqué pour l'affichage ; le téléchargement contient les listes et rayons complets. Aucune image n'est produite.

Le bloc `SLR2` est little endian et de longueur variable :

| Élément | Octets | Contenu |
| --- | ---: | --- |
| En-tête | 12 | `SLR2`, nombre de rayons `uint16`, réservé `uint16`, nombre de retours `uint32` |
| Rayon | 12 | angle et portée `float32`, nombre de retours `uint16`, censure `uint8`, réservé `uint8` |
| Retour | 20 | entrée, sortie, taille `float32`, index d'objet `uint32`, classe `uint8`, méthode `uint8`, réservé `uint16` |

La classe est `0=none`, `1=enemy_head`, `2=enemy_body`, `3=self_body`, `4=border`, `5=food`, `6=prey`. L'index `0xffffffff` représente `-1`, une taille absente est `NaN`, la méthode `0=intersection`, `1=sector`. `format.js` offre `pack`, `inspect` et `unpack`. Le stockage conserve le résultat dans IndexedDB, sans transmission.

## Validation sur une partie réelle

1. Vérifier l'exposition des variables et la synchronisation caméra, position et cap.
2. Comparer des distances tête, corps et bordure à plusieurs zooms ; recalibrer les rayons estimés.
3. Vérifier les valeurs `sz`, `sct`, `fam`, `rank`, `snake_count` et la formule du score affiché.
4. Mesurer CPU et volume par échantillon quand beaucoup de boules et de vers sont visibles ; ajuster la fréquence ou l'indexation spatiale si nécessaire.
5. Contrôler que chaque session ne contient que des états `ok`, des horodatages croissants et des actions humaines.
