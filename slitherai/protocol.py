PROTOCOL = 'common-reference-v2'


def protocol_settings():
    return dict(version=PROTOCOL, anchor_games=4, rotating_games=1,
                anchor_weight=.8, aggregate='half_mean_half_median',
                opponents='fixed_food_and_avoidance', validation_maps=32, validation_seconds=90)
