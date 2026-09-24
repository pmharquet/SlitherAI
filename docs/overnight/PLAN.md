# Nuit du 24 au 25 septembre 2026

## Mandat

Améliorer le modèle NEAT de Slither jusqu'au **25 septembre 2026 à 08:00 Europe/Paris** (06:00 UTC). Début effectif : 18:47 Paris. Priorité à l'apprentissage mesuré et au temps GPU utile. Aucun push, aucune publication. Le superviseur conserve un rôle de contrôle ; analyses et implémentations bornées sont confiées à **GPT-6 Luna, effort xhigh**, avec contexte réduit. Aucun changement du modèle de la conversation principale n'a été effectué par outil.

## Invariants

- Les actions du candidat proviennent exclusivement de son réseau évolué. Aucune aide heuristique, action corrigée, comportement imposé, imitation cachée ou stratégie programmée pour le candidat.
- Les adversaires de référence existants servent à l'évaluation ; leurs scores ne doivent jamais être présentés comme ceux du modèle appris. Toute introduction de self-play doit préserver une épreuve de comparaison indépendante.
- Conserver le contrat 530 entrées et 2 sorties, sauf bénéfice démontré et migration explicitement documentée. NEAT reste le moteur d'apprentissage.
- Préserver les anciens runs, champions et checkpoints. Ne pas mélanger les fitness d'objectifs/protocoles différents. Un changement incompatible crée une nouvelle expérience avec provenance ; ne jamais prétendre à une reprise équivalente.
- Une modification à la fois par hypothèse, avec comparaison sur les mêmes scénarios et budget. Les tests finaux indépendants ne servent pas au réglage.
- Pas de contrôle du jeu officiel, achat, installation inutile, accès aux secrets, modification globale des réglages du PC, ou service exposé au réseau.

## Organisation

1. Audit initial des trajectoires d'apprentissage et paramètres : agent `audit_training`.
2. Audit JSON compact reproductible : agent `audit_tooling`.
3. Performance et fidélité des observations/actions : agent `performance_review`.
4. Le superviseur examine les résultats, attribue un périmètre de fichiers sans chevauchement, relit chaque diff, demande corrections et valide les tests pertinents.
5. Chaque agent d'implémentation commet uniquement ses fichiers, localement : `git add -- <ses fichiers>` puis **`git commit --only -- <ses fichiers>`** pour exclure les fichiers stagés par un autre agent. Pas de `git add .`, pas de push. Ne pas reset/amend/réécrire la branche partagée ; en cas de mélange accidentel, prévenir le superviseur et corriger par un nouveau commit. L'arbre partagé exige de coordonner les commits et toute modification d'un fichier possédé par un autre agent.
6. Un seul responsable des démarrages/arrêts d'entraînement à la fois. Ne pas interrompre une génération pour un changement cosmétique. Les essais GPU concurrents doivent être évités ; réserver une fenêtre de benchmark si nécessaire.

## Boucle de suivi, toutes les 15 minutes

- Automation heartbeat `slitherai-audit-nocturne-15-min`, attachée à cette tâche. Lire STATE.md et le dernier audit, vérifier les agents encore actifs, puis lire une sortie compacte.
- Si aucune génération nouvelle et aucune anomalie : ne pas relire les gros fichiers, ne pas relancer un audit conceptuel ou une recherche web. Laisser tourner et rendre la main.
- Sur nouvelle génération : suivre validation fixe, score fixe moyen, nourriture, survie, causes de mort, boost, diversité, seuil, taille des espèces, complexité et coût par génération. Distinguer bruit, changement de population et progrès généralisable.
- Sur anomalie : déléguer une analyse bornée à Luna xhigh, chercher des sources pertinentes (code officiel/papers/GitHub, Reddit comme pistes anecdotiques), proposer une hypothèse testable, puis revoir avant adoption.
- Si service arrêté : vérifier les processus existants avant de relancer le serveur caché et reprendre le dernier checkpoint compatible via API. Maintenir la cible de génération et les réglages sauvegardés. Ne pas lancer de doublon.
- Ne pas réinitialiser continuellement une population en raison d'un mauvais score isolé. Préférer une reprise compatible ; comparer une branche d'expérience seulement avec une justification et un budget borné.

## Contrôle qualité / preuves

Pour chaque intervention : cause observée, hypothèse, paramètres touchés et interactions, comparaison avant/après, commande de tests, commit, run/checkpoint d'origine, décision retenue/refusée. Vérifier absence de non-finis, déterminisme de reprise si affecté, isolement des arènes et équité d'évaluation si affectés. Ne pas annoncer un gain d'apprentissage sur un simple test logiciel.

Garder un inventaire des paramètres : capteurs/normalisation ; dynamique et densité ; action et récurrence ; initialisation et saturation ; mutations/crossover ; seuil/espèces/stagnation/reproduction ; récompense ; cartes et adversaires ; durée, population, réplications ; validation ; CUDA et coût. Prioriser selon les données et noter les paramètres non encore étudiés.

## Fin de nuit

Vers 07:00–07:30, cesser les expériences risquées et réserver le temps nécessaire à comparer quelques champions sur un jeu final indépendant, avec mêmes cartes et adversaires, graines pré-définies et scores par partie. Conserver le meilleur candidat démontré, les réserves et les incertitudes. À 08:00 : ne plus lancer de changements/expériences ; finaliser le bilan, désactiver le heartbeat et terminer le goal une fois les livrables produits. Préserver toute sauvegarde ; éviter de perdre une génération en cours par arrêt brutal.

Jeu final réservé dès 19:09 : graine **741852963**, **64 arènes indépendantes**, **90 secondes**, adversaires de référence existants et positions focales `(arange(64)*7) % worms`. Ne pas utiliser cette graine pour les entraînements, réglages ou benchmarks préparatoires. Choisir d'abord le candidat principal sur la validation existante, puis comparer au modèle initial et aux contrôleurs de référence sur ce jeu final. Si l'on choisit un autre champion après consultation de ce test, le signaler comme sélection sur le test ; aucune affirmation d'évaluation indépendante du gagnant sans autre jeu tenu à l'écart.

## Économie

Pas de boucle de tokens permanente ni de réveils supplémentaires pour combler l'attente. Rapports bornés, prompts spécialisés, sources ciblées. Luna utilise aussi le quota : ne pas supposer qu'il est gratuit ou illimité. Pas de reset de quota sans confirmation explicite séparée.
