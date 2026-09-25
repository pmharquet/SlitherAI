PROTOCOL = 'common-reference-v2'
SELFPLAY_PROTOCOL = 'population-selfplay-v1'


def protocol_settings(opponent_mode='reference', training_games=5):
    """Describe the selection environment saved with a training run.

    Keep the original dictionary byte-for-byte compatible with existing runs.
    Self-play changes selection fitness, so it must have a distinct version.
    Both modes use the same fixed reference-opponent validation suite.
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
        return dict(version=SELFPLAY_PROTOCOL, opponent_mode='selfplay',
                    games_per_genome=training_games, aggregate='half_mean_half_median_all_games',
                    opponents='co_evolving_population', matchmaking='seat_rotation_cohort_shuffle_v1',
                    validation_opponents='fixed_food_and_avoidance',
                    validation_maps=32, validation_seconds=90)
    raise ValueError(f'Unknown opponent mode: {opponent_mode}')


def recognized_protocol(saved):
    """Return (mode, games) only for an exact supported protocol record."""
    if saved == protocol_settings():
        return 'reference', 5
    if isinstance(saved, dict) and saved.get('opponent_mode') == 'selfplay':
        games = saved.get('games_per_genome')
        try:
            if saved == protocol_settings('selfplay', games):
                return 'selfplay', games
        except ValueError:
            pass
    return None
