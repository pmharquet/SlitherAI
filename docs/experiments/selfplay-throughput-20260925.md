# Expérience de débit et de qualité — 25 septembre 2026

## Constat

L'ancien run `20260924-151316-839796` a enregistré 50 générations en 43 min 34 s, médiane 44 s/génération. Le run de référence `20260924-163112-852097` a enregistré 47 générations, médiane 17,30 min/génération, soit 23,6 fois plus. Le protocole de l'ancien run n'est pas conservé; ses scores et son compte de parties ne sont pas directement comparables. Le nouveau run évalue explicitement 256 génomes × 5 parties privées × 90 s sur des arènes à un apprenant et quinze adversaires fixes. La dernière génération complète est au checkpoint 47, SHA-256 `d7cde85273fa37a6a776a34fd49b6a4a5f2bba304a5c213a5ebc6f331ee29361`. Après ce checkpoint, le service et l'entraîneur ont disparu vers 08:05 sans erreur Python enregistrée; aucune cause système n'est établie.

## Modification testée

Le protocole `population-selfplay-v1` fait jouer les 256 génomes simultanément en 16 arènes × 16 vers, tous pilotés par NEAT. Chaque génome est évalué une fois par jeu; les cohortes sont remélangées et les sièges changent entre jeux. La sélection utilise 50 % de la moyenne et 50 % de la médiane des jeux. Les règles physiques, 530 entrées, 2 sorties et la récompense `growth-v2` sont conservées. Les adversaires de validation restent les contrôleurs de référence sur 32 cartes × 90 s. Les scores de sélection entre les deux protocoles ne sont pas comparables directement.

Le run A `runs/20260925-selfplay-cp47-pilot` part du checkpoint 47, cinq jeux × 90 s, cinq générations, `sensor_chunk=4`. Sa G1 a demandé 127,06 s d'évaluation, soit 8,2 fois moins que la médiane récente d'environ 1 038 s, avec le même nombre de parties par génome. Son premier champion valide à 19,389 sur les 32 cartes de sélection; le meilleur modèle conservé G45 du run de référence reste à 27,438 sur cette suite. Attendre G5 avant de juger l'apprentissage.

## Deuxième budget et règle de comparaison

Après G5 du run A, lancer un run B **distinct** depuis le même checkpoint 47 avec deux jeux × 45 s par génération, `sensor_chunk=4`, 20 générations. Hypothèse de débit : environ 25–40 s/génération, à mesurer; aucun gain de qualité présumé. Les deux runs consommeront un temps GPU du même ordre, mais B donnera davantage d'itérations évolutives et une fitness par génération plus bruitée. Ne changer ni le capteur ni la récompense en même temps. Aucun contrôle heuristique ne pilote un ver candidat dans ces runs.

La suite indépendante de comparaison est réservée maintenant, **avant** de regarder G5 ou B : graine `64821973`, **64 cartes × 90 s**, focal `(arange(64)*7)%worms`, adversaires heuristiques fixes, mêmes paramètres physiques et capteur `legacy-v1`. Comparer en une seule évaluation le champion figé G45, le meilleur validé A et le meilleur validé B; si B n'a pas de champion valide ou échoue techniquement, le documenter plutôt que changer la règle après coup. Ne pas utiliser cette graine pour ajuster les paramètres ou choisir un autre checkpoint ensuite. Les validations32 cartes à seed938271 servent à présélectionner un champion par run; l'épreuve indépendante sert à juger la généralisation, avec incertitude appariée. Le protocole self-play peut produire des cycles de coévolution; une hausse du score de sélection sans gain sur adversaires fixes ne sera pas comptée comme un progrès.

## Vérification préalable

- Tests ciblés CPU : 54 réussis, couvrant mapping slot→génome, parité de la simulation manuelle courte, score, isolation de protocole, reprise, warm-start, holdout et compatibilité de référence.
- Smoke CUDA : 256 sorties finies sur 16 arènes × 16 vers, une étape.
- Tableau de bord local : `http://127.0.0.1:8765/`, run A affiché; le serveur connaît et peut reprendre le protocole versionné.
- Code du protocole : commit local `d6dd0d3`; aucun push.

## Résultats A/B et épreuve réservée

Le run A a terminé 5 générations : 781,84 s d'évaluation cumulée, soit 156,37 s/génération en moyenne. Son meilleur validé reste le génome initial 8808 (19,389 sur les 32 cartes de sélection); la validation G5 vaut 14,210. Le run B a terminé 20 générations en 944,6 s de temps réel, soit 47,23 s/génération avec les validations comprises. Son évaluation prend 32,00 s/génération en moyenne. Ses validations G1/G5/G10/G15/G20 valent respectivement 19,389 / 16,971 / 6,811 / 17,883 / 8,236. Son meilleur validé est aussi le génome initial 8808. Les fichiers pickle A/B diffèrent par leurs métadonnées, mais contiennent le même génome (SHA-256 canonique `9dcee8a5d57568b853822f1648b7f450a48fcd74c7efd2446a390f1cb8421142`).

La suite indépendante pré-déclarée seed `64821973`, 64 cartes × 90 s a été exécutée une fois après les deux pilotes. Résultat : G45 = **24,387 ± 1,925** (erreur type), A = B = initial = **19,571 ± 1,792**. Différence appariée G45 − A/B = **+4,816**, intervalle bootstrap 95 % calculé [0,674 ; 8,844] pour G45 contre initial. Les sorties détaillées et les sources figées sont dans `runs/20260925-selfplay-comparison/`. L'auto-jeu pur a retrouvé le débit ancien mais n'a pas amélioré le meilleur modèle. Le G45 figé reste le modèle à conserver. Cette suite est maintenant consommée et ne doit pas servir à régler C.

## Expérience C pré-déclarée

Le prochain pilote testera l'évaluation de plusieurs candidats dans une arène avec des adversaires de référence fixes. Le modèle de départ sera `checkpoint-44` du run lent, qui contient le génome G45 `8270`. Vérification ultérieure : ce même génome est aussi présent et inchangé dans les checkpoints 45–47, y compris la source des pilotes A/B. Leur sélection a retenu un autre champion au premier jeu; elle n'a pas perdu le génome G45 dans le checkpoint. Le choix de `checkpoint-44` est conservé pour le pilote C pré-déclaré, sans supposer qu'il est supérieur à `checkpoint-47`. Config prévue : population 256, 32 cartes × 16 vers, 8 candidats NEAT et 8 adversaires fixes par carte, chaque candidat une fois par jeu, 2 jeux × 45 s/génération, 10 générations, capteur `legacy-v1`, récompense `growth-v2`, validation 32 cartes × 90 s aux G1/G5/G10. Le débit et la validation seront comparés aux résultats précédents sans prétendre comparer directement les scores de sélection entre protocoles. Si une amélioration apparaît, une nouvelle suite indépendante est réservée maintenant : seed `91837261`, 64 cartes × 90 s, même configuration physique, confrontation appariée G45 contre le meilleur C figé. Aucune épreuve indépendante ne sera répétée pour choisir un autre checkpoint.
