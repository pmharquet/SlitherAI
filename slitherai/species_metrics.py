"""Read-only NEAT telemetry; selection and stagnation remain owned by NEAT."""
import gzip
import math
import pickle
import statistics
from pathlib import Path
import neat
from .io import read_json, write_json


def composition(species, generation):
    rows = [dict(id=s.key, size=len(s.members), created=s.created,
                 age=max(0, generation-s.created), last_improved=s.last_improved)
            for s in sorted(species.species.values(), key=lambda s: s.key)]
    total = sum(s['size'] for s in rows)
    shares = [s['size']/total for s in rows] if total else []
    return dict(generation=generation, population=total, rows=rows, count=len(rows),
                effective=math.exp(-sum(p*math.log(p) for p in shares if p)),
                largest_share=max(shares, default=0))


class SpeciesReporter(neat.reporting.BaseReporter):
    def __init__(self, run, population):
        self.run = Path(run)
        self.population = population
        config = population.config
        self.parameters = dict(compatibility_threshold=config.species_set_config.compatibility_threshold,
            max_stagnation=config.stagnation_config.max_stagnation,
            species_elitism=config.stagnation_config.species_elitism,
            species_fitness_func=config.stagnation_config.species_fitness_func,
            elitism=config.reproduction_config.elitism,
            survival_threshold=config.reproduction_config.survival_threshold)
        for key in ('progress_window', 'progress_delta', 'max_removals'):
            self.parameters[key] = getattr(config.stagnation_config, key, None)
        for key in ('target_min', 'target_max'):
            self.parameters[key] = getattr(config.species_set_config, key, None)
        self.history = read_json(self.run / 'species-history.json', [])
        self.history = [r for r in self.history if r['generation'] < population.generation]
        saved = read_json(self.run / 'species.json', {})
        self.latest = saved.get('latest')
        if self.latest and self.latest['generation'] >= population.generation:
            self.latest = None
        self.current = composition(population.species, population.generation)
        self.generation = population.generation
        self.backfill_composition()

    def backfill_composition(self):
        """Old checkpoints reliably expose membership, but not all evaluated fitnesses."""
        known = {r['generation'] for r in self.history}
        for path in sorted(self.run.glob('checkpoint-*')):
            try:
                generation = int(path.name.split('-')[-1])
            except ValueError:
                continue
            if generation >= self.population.generation or generation in known:
                continue
            with gzip.open(path, 'rb') as stream:
                generation, _, _, species, _ = pickle.load(stream)
            row = composition(species, generation)
            row['reconstructed'] = True
            self.history.append(row)
        self.history.sort(key=lambda row: row['generation'])
        write_json(self.run / 'species-history.json', self.history)

    def publish(self):
        self.parameters['compatibility_threshold'] = self.population.config.species_set_config.compatibility_threshold
        write_json(self.run / 'species.json', dict(current=self.current, latest=self.latest,
                                                  parameters=self.parameters))

    def start_generation(self, generation):
        self.generation = generation
        self.current = composition(self.population.species, generation)
        self.removed = set()
        self.extinction = False
        self.publish()

    def post_evaluate(self, config, population, species, best_genome):
        self.evaluated = dict(self.current, rows=[])
        self.references = dict(species.species)
        # Copy scores before reproduce clears members and speciate reassigns them.
        for sid, s in sorted(self.references.items()):
            members = list(s.members.values())
            scores = [float(g.fitness) for g in members]
            self.evaluated['rows'].append(dict(id=sid, size=len(members), created=s.created,
                age=max(0, self.generation-s.created), best=max(scores), mean=statistics.mean(scores),
                anchor_mean=statistics.mean(getattr(g, 'anchor_fitness', g.fitness) for g in members),
                food_gain=statistics.mean(getattr(g, 'behavior', {}).get('food_gain', 0.) for g in members),
                survival=statistics.mean(getattr(g, 'behavior', {}).get('alive', 0.) for g in members),
                hidden=statistics.mean(sum(k not in config.genome_config.output_keys for k in g.nodes) for g in members),
                connections=statistics.mean(sum(c.enabled for c in g.connections.values()) for g in members)))

    def species_stagnant(self, sid, species):
        self.removed.add(sid)

    def complete_extinction(self):
        self.extinction = True

    def end_generation(self, config, population, species_set):
        row = self.evaluated
        # These are the actual updated values used by DefaultStagnation/Reproduction.
        protected = sorted(self.references, key=lambda sid: (self.references[sid].fitness, sid))
        elite = self.parameters['species_elitism']
        protected = set(protected[-elite:]) if elite else set()
        for s in row['rows']:
            ref = self.references[s['id']]
            s.update(last_improved=ref.last_improved, stagnant_for=max(0, self.generation-ref.last_improved),
                recent_score=getattr(ref, 'recent_score', None),
                adjusted_fitness=ref.adjusted_fitness, protected=s['id'] in protected,
                removed=s['id'] in self.removed,
                next_size=len(species_set.species[s['id']].members) if s['id'] in species_set.species else 0)
        row.update(new_ids=sorted(set(species_set.species)-set(self.references)),
                   removed_ids=sorted(set(self.references)-set(species_set.species)),
                   complete_extinction=self.extinction, reconstructed=False)
        self.latest = row
        self.history = [r for r in self.history if r['generation'] != self.generation] + [row]
        self.history.sort(key=lambda r: r['generation'])
        write_json(self.run / 'species-history.json', self.history)
        self.current = composition(species_set, self.generation+1)
        self.publish()
