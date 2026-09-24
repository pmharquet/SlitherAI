"""Versioned objectives, also usable to rescore recorded episode outcomes."""
import torch

REWARD_VERSION = 'growth-v2'


def reward_terms(gained, spent, age, kills, alive, version=REWARD_VERSION):
    if version == 'legacy-v1':
        return dict(food=gained*.4, boost=-spent*.6, survival=age*.2,
                    kills=kills*50, death=(~alive)*-40)
    if version != REWARD_VERSION:
        raise ValueError(f'Unknown reward version: {version}')
    # Nearly linear for small gains, logarithmic for very large gains: one lucky
    # corpse does not dominate all of a genome's other episodes. Net growth
    # also prevents earning food reward by re-eating one's own boost deposits.
    return dict(growth=25*torch.asinh((gained-spent)/25), survival=age*.02,
                kills=2*torch.sqrt(kills), death=(~alive)*-12)


def reward_score(gained, spent, age, kills, alive, version=REWARD_VERSION):
    return sum(reward_terms(gained, spent, age, kills, alive, version).values())
