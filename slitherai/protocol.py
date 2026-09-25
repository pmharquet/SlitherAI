PROTOCOL = 'common-reference-v2'
SELFPLAY_PROTOCOL_V1 = 'population-selfplay-v1'
SELFPLAY_PROTOCOL = 'population-selfplay-v2'
SELFPLAY_STAGNATION_WINDOW = 5
SELFPLAY_STAGNATION_DELTA = .02
MIXED_REFERENCE_PROTOCOL_V1 = 'mixed-reference-v1'
MIXED_REFERENCE_PROTOCOL_V2 = 'mixed-reference-v2'
MIXED_REFERENCE_PROTOCOL_V3 = 'mixed-reference-v3'
MIXED_REFERENCE_PROTOCOL_V4 = 'mixed-reference-v4'
MIXED_REFERENCE_PROTOCOL = 'mixed-reference-v5'


def protocol_settings(opponent_mode='reference', training_games=5, selfplay_version=None,
                      mixed_version=None):
    """Describe the selection environment saved with a training run.

    Keep the original dictionary byte-for-byte compatible with existing runs.
    Self-play v2 keeps raw selection fitness and tracks stagnation using
    within-generation midrank percentiles. Mixed-reference versions preserve
    their complete matchup and duration configuration for strict resume.
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
        if mixed_version is None:
            version = {1: 5, 2: 3, 4: 4}.get(training_games)
            if version is None:
                raise ValueError('Mixed-reference requires exactly one, two or four training games')
        else:
            version = mixed_version
        if version not in (1, 2, 3, 4, 5):
            raise ValueError(f'Unsupported mixed-reference protocol version: {version}')
        expected_games = {4: 4, 5: 1}.get(version, 2)
        if training_games != expected_games:
            raise ValueError(f'Mixed-reference v{version} requires exactly {expected_games} training games')
        is_v4 = version == 4
        is_v5 = version == 5
        return dict(version=({1: MIXED_REFERENCE_PROTOCOL_V1,
                              2: MIXED_REFERENCE_PROTOCOL_V2,
                              3: MIXED_REFERENCE_PROTOCOL_V3,
                              4: MIXED_REFERENCE_PROTOCOL_V4,
                              5: MIXED_REFERENCE_PROTOCOL}[version]),
                    opponent_mode='mixed-reference', population=256,
                    maps_per_game=(256 if is_v5 else 64 if version in (3, 4) else 32), worms_per_map=16,
                    candidate_slots_per_map=(1 if is_v5 else 4 if version in (3, 4) else 8),
                    reference_slots_per_map=(15 if is_v5 else 12 if version in (3, 4) else 8),
                    games_per_genome=expected_games, seconds_per_game=(45 if version == 1 else 90),
                    aggregate=('single_game_score' if is_v5 else
                               'arithmetic_mean_four_games' if is_v4
                               else 'arithmetic_mean_two_games'),
                    opponents='fixed_heuristic_policy',
                    matchmaking=('seeded_genome_shuffle_generation_seat_rotation_v1'
                                 if is_v5 else
                                 'cohort_shuffle_four_block_sixteen_seat_rotation_v1'
                                 if is_v4 else
                                 'cohort_shuffle_four_seat_generation_rotation_v1'
                                 if version == 3 else 'cohort_shuffle_eight_seat_rotation_v1'),
                    **({'sensor_chunk': 8} if version in (3, 4, 5) else {}),
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
            for version in (1, 2, 3, 4, 5):
                games = {4: 4, 5: 1}.get(version, 2)
                if saved == protocol_settings('mixed-reference', games, mixed_version=version):
                    return 'mixed-reference', games
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
    for version in (1, 2, 3, 4, 5):
        try:
            games = {4: 4, 5: 1}.get(version, 2)
            if saved == protocol_settings('mixed-reference', games, mixed_version=version):
                return version
        except ValueError:
            pass
    return None
