PROTOCOL = 'common-reference-v2'
SELFPLAY_PROTOCOL_V1 = 'population-selfplay-v1'
SELFPLAY_PROTOCOL = 'population-selfplay-v2'
SELFPLAY_STAGNATION_WINDOW = 5
SELFPLAY_STAGNATION_DELTA = .02
MIXED_REFERENCE_PROTOCOL = 'mixed-reference-v1'


def protocol_settings(opponent_mode='reference', training_games=5, selfplay_version=None,
                      mixed_version=None):
    """Describe the selection environment saved with a training run.

    Keep the original dictionary byte-for-byte compatible with existing runs.
    Self-play v2 keeps raw selection fitness and tracks stagnation using
    within-generation midrank percentiles. Version 1 remains available for
    exact resume compatibility with existing runs.
    """
    if opponent_mode == 'reference':
        if training_games != 5:
            raise ValueError('The reference protocol requires five games')
        return dict(version=PROTOCOL, anchor_games=4, rotating_games=1,
                    anchor_weight=.8, aggregate='half_mean_half_median',
                    opponents='fixed_food_and_avoidance', validation_maps=32, validation_seconds=90)
    if opponent_mode == 'selfplay':
        if not isinstance(training_games, int) or not 1 <= training_games <= 16:
            raise ValueError('Self-play requires 1 to 16 games per genome')
        version = 2 if selfplay_version is None else selfplay_version
        if version == 1:
            return dict(version=SELFPLAY_PROTOCOL_V1, opponent_mode='selfplay',
                    games_per_genome=training_games, aggregate='half_mean_half_median_all_games',
                    opponents='co_evolving_population', matchmaking='seat_rotation_cohort_shuffle_v1',
                    validation_opponents='fixed_food_and_avoidance',
                    validation_maps=32, validation_seconds=90)
        if version != 2:
            raise ValueError(f'Unsupported self-play protocol version: {version}')
        return dict(version=SELFPLAY_PROTOCOL, opponent_mode='selfplay',
                    games_per_genome=training_games, aggregate='half_mean_half_median_all_games',
                    opponents='co_evolving_population', matchmaking='seat_rotation_cohort_shuffle_v1',
                    stagnation_metric='within_generation_midrank_percentile',
                    stagnation_window=SELFPLAY_STAGNATION_WINDOW,
                    stagnation_delta=SELFPLAY_STAGNATION_DELTA,
                    validation_opponents='fixed_food_and_avoidance',
                    validation_maps=32, validation_seconds=90)
    if opponent_mode == 'mixed-reference':
        if training_games != 2:
            raise ValueError('Mixed-reference requires exactly two training games')
        version = 1 if mixed_version is None else mixed_version
        if version != 1:
            raise ValueError(f'Unsupported mixed-reference protocol version: {version}')
        return dict(version=MIXED_REFERENCE_PROTOCOL, opponent_mode='mixed-reference',
                    population=256, maps_per_game=32, worms_per_map=16,
                    candidate_slots_per_map=8, reference_slots_per_map=8,
                    games_per_genome=2, seconds_per_game=45,
                    aggregate='arithmetic_mean_two_games',
                    opponents='fixed_heuristic_policy',
                    matchmaking='cohort_shuffle_eight_seat_rotation_v1',
                    stagnation_metric='within_generation_midrank_percentile',
                    stagnation_window=SELFPLAY_STAGNATION_WINDOW,
                    stagnation_delta=SELFPLAY_STAGNATION_DELTA,
                    validation_opponents='fixed_food_and_avoidance',
                    validation_maps=32, validation_seconds=90,
                    validation_every=5)
    raise ValueError(f'Unknown opponent mode: {opponent_mode}')


def recognized_protocol(saved):
    """Return (mode, games) only for an exact supported protocol record."""
    if saved == protocol_settings():
        return 'reference', 5
    if isinstance(saved, dict) and saved.get('opponent_mode') == 'selfplay':
        games = saved.get('games_per_genome')
        try:
            if saved == protocol_settings('selfplay', games, selfplay_version=1):
                return 'selfplay', games
            if saved == protocol_settings('selfplay', games, selfplay_version=2):
                return 'selfplay', games
        except ValueError:
            pass
    if isinstance(saved, dict) and saved.get('opponent_mode') == 'mixed-reference':
        try:
            if saved == protocol_settings('mixed-reference', 2, mixed_version=1):
                return 'mixed-reference', 2
        except ValueError:
            pass
    return None


def selfplay_protocol_version(saved):
    """Return the exact supported self-play version, else None."""
    if not isinstance(saved, dict) or saved.get('opponent_mode') != 'selfplay':
        return None
    games = saved.get('games_per_genome')
    try:
        for version in (1, 2):
            if saved == protocol_settings('selfplay', games, selfplay_version=version):
                return version
    except ValueError:
        pass
    return None


def mixed_reference_protocol_version(saved):
    """Return the exact supported mixed-reference version, else None."""
    try:
        return 1 if saved == protocol_settings('mixed-reference', 2, mixed_version=1) else None
    except ValueError:
        return None
