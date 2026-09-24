"""NEAT policies for noisy games: bounded speciation feedback and recent progress."""
import statistics
import neat
from neat.config import ConfigParameter, DefaultClassConfig
from neat.species import Species, GenomeDistanceCache


class AdaptiveSpeciesSet(neat.DefaultSpeciesSet):
    @classmethod
    def parse_config(cls, values):
        return DefaultClassConfig(values, [
            ConfigParameter('compatibility_threshold', float, 1.9),
            ConfigParameter('target_min', int, 8), ConfigParameter('target_max', int, 12),
            ConfigParameter('threshold_step', float, .08),
            ConfigParameter('threshold_min', float, .3), ConfigParameter('threshold_max', float, 4.)])

    def speciate(self, config, population, generation):
        c = self.species_set_config
        minimum = max(config.reproduction_config.min_species_size, config.reproduction_config.elitism, 1)
        capacity = max(1, len(population)//minimum)
        low, high = min(c.target_min, capacity), min(c.target_max, capacity)
        if not 1 <= low <= high or c.threshold_step <= 0:
            raise ValueError('Invalid adaptive species settings')
        distances = GenomeDistanceCache(config.genome_config)
        keys = sorted(population)
        if not self.species:
            # Calibrate the initial threshold against this population rather than
            # assuming a distance scale from another task or activation range.
            def count_at(threshold):
                reps = []
                for key in keys:
                    if not reps or min(distances(population[key], population[r]) for r in reps) >= threshold:
                        reps.append(key)
                return len(reps)
            left, right = c.threshold_min, c.threshold_max
            best = (float('inf'), c.compatibility_threshold)
            for _ in range(12):
                threshold = (left+right)/2
                count = count_at(threshold)
                loss = max(low-count, count-high, 0)
                best = min(best, (loss, threshold))
                if loss == 0:
                    break
                if count > high: left = threshold
                else: right = threshold
            c.compatibility_threshold = best[1]
        else:
            change = -c.threshold_step if len(self.species) < low else c.threshold_step if len(self.species) > high else 0
            c.compatibility_threshold = min(c.threshold_max, max(c.threshold_min, c.compatibility_threshold+change))

        remaining = list(keys)
        representatives, members = {}, {}
        for sid, species in sorted(self.species.items()):
            representative = min(remaining, key=lambda k: (distances(species.representative, population[k]), k))
            representatives[sid], members[sid] = representative, [representative]
            remaining.remove(representative)
        for key in remaining:
            choices = sorted((distances(population[key], population[r]), sid) for sid, r in representatives.items())
            if choices and (choices[0][0] < c.compatibility_threshold or len(representatives) >= capacity):
                sid = choices[0][1]
            else:
                sid = next(self.indexer)
                representatives[sid], members[sid] = key, []
            members[sid].append(key)
        self.genome_to_species = {}
        updated = {}
        for sid, keys in members.items():
            species = self.species.get(sid) or Species(sid, generation)
            species.update(population[representatives[sid]], {k:population[k] for k in keys})
            updated[sid] = species
            self.genome_to_species.update({key:sid for key in keys})
        self.species = updated


class WindowedStagnation(neat.DefaultStagnation):
    @classmethod
    def parse_config(cls, values):
        return DefaultClassConfig(values, [
            ConfigParameter('species_fitness_func', str, 'mean'),
            ConfigParameter('max_stagnation', int, 30), ConfigParameter('species_elitism', int, 4),
            ConfigParameter('progress_window', int, 5), ConfigParameter('progress_delta', float, .25),
            ConfigParameter('max_removals', int, 1)])

    def update(self, species_set, generation):
        c = self.stagnation_config
        if c.progress_window < 2 or c.max_removals < 1 or c.species_fitness_func != 'mean':
            raise ValueError('Windowed stagnation requires mean fitness and a window >= 2')
        ranked = []
        for sid, species in sorted(species_set.species.items()):
            # Anchor scores use unchanged scenarios/opponents. The changing
            # generalization episode does not reset or expire stagnation records.
            species.fitness = statistics.mean(getattr(g, 'anchor_fitness', g.fitness) for g in species.members.values())
            species.adjusted_fitness = None
            species.fitness_history.append(species.fitness)
            history, window = species.fitness_history, c.progress_window
            species.recent_score = statistics.median(history[-window:])
            species.progress_delta = None
            if len(history) == window:
                species.last_improved = generation
            elif len(history) >= 2*window and len(history) % window == 0:
                previous = statistics.median(history[-2*window:-window])
                species.progress_delta = species.recent_score-previous
                if species.progress_delta > c.progress_delta:
                    species.last_improved = generation
            ranked.append((sid, species))
        ranked.sort(key=lambda item: (item[1].fitness, item[0]))
        protected = {sid for sid, _ in ranked[-c.species_elitism:]} if c.species_elitism else set()
        remaining, removed, result = len(ranked), 0, []
        for sid, species in ranked:
            eligible = generation-species.last_improved >= c.max_stagnation
            stagnant = (eligible and sid not in protected and remaining > max(1, c.species_elitism)
                        and removed < c.max_removals)
            species.stagnation_eligible = eligible
            species.protected = sid in protected
            if stagnant:
                remaining -= 1
                removed += 1
            result.append((sid, species, stagnant))
        return result
