# État de la nuit

Mise à jour : 24 septembre 2026, environ 18:50 Europe/Paris.

- Repo Git existant, point de départ propre `3094b71` (Initial public release).
- Service initialement arrêté ; relancé caché sur http://127.0.0.1:8765/.
- Reprise demandée avec succès du run `runs/20260924-163112-852097`, dernier checkpoint 3, pour 47 générations supplémentaires (cible totale 50), CUDA.
- Récompense growth-v2, protocole common-reference-v2, 256 réseaux, 64 arènes × 16 vers, 5 parties/réseau, 90 secondes. Ne pas modifier à l'aveugle.
- Heartbeat créé : `slitherai-audit-nocturne-15-min`, toutes les 15 minutes ; échéance logique 25 septembre 08:00 Europe/Paris.
- Bilan final planifié à 08:00 : heartbeat `slitherai-bilan-08-h` (une occurrence).
- Agents actifs : audit_training (rapport données/recherches), audit_tooling (CLI audit et tests, autorisé à commettre ses nouveaux fichiers), performance_review (audit CPU/GPU/capteurs en lecture seule).
- Prochaine action : vérifier reprise active, attendre les conclusions et examiner les premiers commits. Aucun patch de dynamique d'entraînement appliqué pour la nuit à ce stade.
- Le graphique demandé dans le chat a été livré précédemment ; aucune modification du tableau de bord n'est nécessaire pour cette demande.

## Journal des décisions

- 18:50 : reprendre le modèle existant pendant les audits pour conserver le temps GPU et établir une référence.
- Reprise vérifiée : actif, génération interne 3 (G4 affichée), cible 50, aucune erreur. 38 espèces, seuil 2,0209375.
- Audit initial préliminaire : nourriture moyenne progresse de 7,4 à 10,8 sur G1–G3, mais survie finale ~0,31 % ; collisions bord dominantes. Le nombre d'espèces 8→30→36→38 nécessite un diagnostic des distances et des pools de parents avant correction. Agent audit_training chargé de cette preuve CPU. Conserver la référence au moins jusqu'à la validation G5 si aucune erreur bloquante.
- Agent performance_review autorisé à une seule pause API de **3 minutes maximum**, restauration du champ pause dans finally, sans arrêt/reprise du processus, pour comparer sensor_chunk 4/8/16 avec égalité des observations. Il est seul opérateur des contrôles pendant cette fenêtre. Pas encore de patch du simulateur autorisé.
- Fenêtre terminée vers 18:55, pause=false restauré et phase training confirmée. Microbenchmark monde jeune : observation chunk4/8/16 = 43,17/24,77/15,03 ms ; observation+step = 55,54/36,82/28,00 ms ; sorties bit-à-bit identiques. Ce résultat ne prouve pas encore un gain sur génération complète ; demander couverture corps longs/morts et confirmation après checkpoint G5 avant adoption.
- Revue du brouillon overnight_audit : corrections demandées à l'agent sur score de sélection (history.mean, distinct de evaluation.fitness brut), phases completed/interrupted, et estimation temporelle (pauses incluses dans durée mesurée ; médiane = estimation, pas borne mathématique). Attendre commit corrigé et tests avant usage décisionnel.
- Prochain point utile : retour des agents / heartbeat prévu ; aucune nouvelle analyse nécessaire avant ces éléments ou une nouvelle génération. Aucun changement du modèle appris déployé.
- 18:57 : commits agents `0ac43b9` (audit JSON) et `b64f77d` (benchmark observation) relus et acceptés. Vérification superviseur : `python -m pytest tests/test_overnight_audit.py tests/test_observation_benchmark.py -q` → **4 passed**. Aucun changement des actions ni du simulateur de production.
- Audit réutilisable : `.venv/Scripts/python.exe -m slitherai.overnight_audit --run runs/20260924-163112-852097 --deadline 2026-09-25T08:00:00+02:00`. Écrit `analysis/overnight/audit.json` et `audit.md` dans le run ; ne lit aucun pickle ni ne contrôle le processus. Confirmer l'activité avec `/api/state` quand une décision l'exige.
- Nouvelles tâches bornées : performance_review prépare une comparaison de politiques complète chunk4/16 + cas CUDA corps longs/morts, sans nouvelle fenêtre GPU encore autorisée ; audit_tooling prépare l'inventaire exhaustif des paramètres (`parameter-map.md`) en lecture seule. audit_training termine le diagnostic des distances de spéciation. Préserver entraînement jusqu'au point de comparaison G5.
- 19:03 : G4 terminée, moyenne sélection −2,7105, meilleur 22,772 ; G5 en cours. Validation comparable G5 encore attendue, pas de preuve de généralisation nouvelle.
- Diagnostic causal de audit_training : reproduction G1 rejouée exactement sur les 256 génomes depuis checkpoint/RNG. 21 des 22 nouvelles espèces sont liées à l'ajout d'un neurone poussant des distances déjà proches du seuil ; la protection de ces innovations est cohérente avec NEAT. **Ne pas forcer une fusion ni relever le seuil uniquement pour afficher 8–12 espèces.** Étudier l'allocation minimale des descendants (4 vs2, élitisme2) avec quantification des niches éventuellement gelées à2 élites avant tout essai borné après G5.
- Hypothèse min_species_size=2 rejetée pour la population actuelle : allocation G4 calculée à partir des mêmes fitness donne 74 élites +182 enfants avec les deux réglages, aucune espèce au plancher ; bénéfice attendu nul. Pas de branche lancée.
- Rejouabilité à éclaircir : G1 est exact, mais replay manuel des transitions suivantes de l'agent diffère sur 33 génomes uniquement pour clés structurelles malgré poids/biais/RNG concordants. Ne pas conclure à un défaut de production ni au hash Python sans reproduire le chemin complet Population.run. Agent chargé d'un diagnostic minimal après rapport.
- Jeu final indépendant réservé dans PLAN.md (graine741852963,64cartes90s) ; ne pas le consommer lors des réglages.

## Livrables attendus

- `docs/overnight/initial-audit.md`
- `docs/overnight/audit-tool.md` + audit exécutable
- `docs/overnight/performance-review.md`
- Journal d'expériences et rapport final au matin, avec provenance du meilleur modèle.
