import neat

from slitherai.config import SimConfig
from slitherai.network import load_config
from scripts.profile_episode_components import COMPONENTS, profile_episode


def test_component_profiler_reports_cpu_phases_and_separate_episode_total():
    neat_config = load_config(4)
    genome = next(iter(neat.Population(neat_config).population.values()))
    config = SimConfig(maps=2, worms=2, foods=8, preys=0, body_points=8,
                       arena_radius=1200, substeps=1, sensor_chunk=4)

    result = profile_episode(
        genome, neat_config, config, seed=71, seconds=.2, device='cpu',
        candidates_per_map=1, warmup_episodes=0, repeats=1)

    assert result['warmup_episodes'] == 0
    assert result['device'] == 'cpu'
    assert result['component_median_seconds'].keys() == set(COMPONENTS)
    assert all(value >= 0 for value in result['component_median_seconds'].values())
    assert result['unprofiled_total_median_seconds'] >= 0
    assert result['profiled_component_sum_median_seconds'] == sum(
        result['component_median_seconds'].values())
    assert result['samples'][0]['profiled']['steps'] == 2
    assert result['samples'][0]['unprofiled']['steps'] == 2
