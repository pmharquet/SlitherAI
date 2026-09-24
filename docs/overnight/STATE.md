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

## Livrables attendus

- `docs/overnight/initial-audit.md`
- `docs/overnight/audit-tool.md` + audit exécutable
- `docs/overnight/performance-review.md`
- Journal d'expériences et rapport final au matin, avec provenance du meilleur modèle.
