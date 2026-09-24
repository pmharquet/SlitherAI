from dataclasses import dataclass

@dataclass
class SimConfig:
    maps: int = 64
    worms: int = 16
    foods: int = 1024
    preys: int = 8
    body_points: int = 96
    arena_radius: float = 2400.
    arena_variation: float = .2
    view_half_width: float = 600.
    view_half_height: float = 500.
    dt: float = .1
    substeps: int = 3
    base_speed: float = 115.8
    boost_speed: float = 210.
    raw_speed_scale: float = 20.
    turn_rate: float = 2.8
    initial_mass: float = 35.
    min_mass: float = 10.
    boost_cost: float = 3.
    body_spacing: float = 9.
    sensor_chunk: int = 4
    sensor_version: str = 'legacy-v1'
    reward_version: str = 'growth-v2'

    @classmethod
    def from_dict(cls, data):
        # Old runs must never silently acquire a new objective.
        return cls(**dict(data, reward_version=data.get('reward_version', 'legacy-v1')))

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'reward_version' not in state:
            self.reward_version = 'legacy-v1'
        if 'sensor_version' not in state:
            self.sensor_version = 'legacy-v1'

    def validate(self):
        for name in ('maps', 'worms', 'foods', 'body_points', 'substeps', 'sensor_chunk'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.sensor_chunk not in (4, 8, 16):
            raise ValueError('sensor_chunk must be one of 4, 8, or 16')
        if self.worms < 2 or self.body_points < 8:
            raise ValueError('Need >=2 worms and >=8 body points')
        if not 0 <= self.preys <= self.foods:
            raise ValueError('preys must be between zero and ambient food count')
        if self.arena_radius < 300 or self.dt <= 0 or self.arena_variation < 0 or self.arena_variation >= .8:
            raise ValueError('Invalid arena or timestep')
        if self.reward_version not in ('legacy-v1', 'growth-v2'):
            raise ValueError('Unknown reward version')
        if self.sensor_version not in ('legacy-v1', 'export-v1'):
            raise ValueError('Unknown sensor version')
        return self
