import math
from dataclasses import asdict
import torch
from .config import SimConfig
from .geometry import point_segment_distance2, ray_circle, ray_capsule
from .schema import ANGLES, INPUTS
from .rewards import reward_score, reward_terms

class WorldBatch:
    """Independent arenas in tensors; no cross-arena interaction. Units are world units/seconds.

    Bodies are arc-length-resampled polylines. Own-body collision is allowed, as in Slither.
    All visible body capsules participate in sensing; spatial culling has no fixed top-K cap.
    """
    def __init__(self, config: SimConfig, device='cpu', seed=1, shared_random=False):
        self.c = config.validate()
        self.device = torch.device(device)
        self.shared_random = shared_random
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.angles = torch.as_tensor(ANGLES, device=self.device)
        self.edges = (self.angles[1:] + self.angles[:-1]) / 2
        self.reset()

    def rand(self, *shape):
        if self.shared_random and shape[0] == self.c.maps:
            # Identical exogenous randomness in each candidate's private arena.
            return torch.rand(1, *shape[1:], device=self.device, generator=self.rng).expand(*shape)
        return torch.rand(*shape, device=self.device, generator=self.rng)

    def random_food(self, count):
        c = self.c
        angle = self.rand(c.maps, count) * math.tau
        radius = self.rand(c.maps, count).sqrt() * self.arena[:, None] * .98
        points = torch.stack((angle.cos(), angle.sin()), -1) * radius[..., None]
        if hasattr(self, 'hotspots'):
            chosen = (self.rand(c.maps, count) * self.hotspots.shape[1]).long()
            centers = self.hotspots.gather(1, chosen[..., None].expand(-1, -1, 2))
            cluster = centers + (self.rand(c.maps, count, 2) - .5) * 280
            points = torch.where((self.rand(c.maps, count) < .3)[..., None], cluster, points)
        return points

    def reset(self):
        c, d = self.c, self.device
        m, w, p = c.maps, c.worms, c.body_points
        self.arena = c.arena_radius * (1 + (self.rand(m) * 2 - 1) * c.arena_variation)
        # Persistent food concentrations differ between independent arenas.
        self.hotspots = (self.rand(m, 8, 2) - .5) * self.arena[:, None, None]
        # A jittered grid prevents unavoidable spawn overlaps.
        side = math.ceil(math.sqrt(w))
        yy, xx = torch.meshgrid(torch.linspace(-.5, .5, side, device=d), torch.linspace(-.5, .5, side, device=d), indexing='ij')
        grid = torch.stack((xx.flatten()[:w], yy.flatten()[:w]), -1)
        self.head = grid[None] * self.arena[:, None, None] + (self.rand(m, w, 2) - .5) * 40
        self.heading = self.rand(m, w) * math.tau
        self.target = self.heading.clone()
        self.mass = c.initial_mass * (.8 + .4 * self.rand(m, w))
        self.initial_mass = self.mass.clone()
        self.alive = torch.ones((m, w), dtype=torch.bool, device=d)
        self.boost = torch.zeros_like(self.alive)
        self.speed = torch.full((m, w), c.base_speed, device=d)
        direction = torch.stack((self.heading.cos(), self.heading.sin()), -1)
        self.body = self.head[:, :, None, :] - direction[:, :, None, :] * torch.arange(p, device=d)[None, None, :, None] * c.body_spacing
        self.age = torch.zeros((m, w), device=d)
        self.gained = torch.zeros_like(self.age)
        self.spent = torch.zeros_like(self.age)
        self.kills = torch.zeros_like(self.age)
        self.decisions = torch.zeros_like(self.age)
        self.boost_steps = torch.zeros_like(self.age)
        self.turn_sum = torch.zeros_like(self.age)
        self.border_deaths = torch.zeros_like(self.alive)
        self.collision_deaths = torch.zeros_like(self.alive)
        self.elapsed = 0.
        self.steps = 0
        self.food = torch.cat((self.random_food(c.foods), torch.zeros((m, w*p, 2), device=d)), 1)
        self.food_mass = torch.cat((.5 + self.rand(m, c.foods) * 2.5, torch.zeros((m, w*p), device=d)), 1)
        self.food_size = (self.food_mass * 3).clamp(2, 20)
        self.shed_cursor = 0
        self.prey_heading = self.rand(m, c.preys) * math.tau

    @property
    def radius(self):
        return (14.5 * (1 + self.mass / 300).pow(.4)).clamp_max(80.)

    @property
    def segments(self):
        return 2 + self.mass / 15.

    @property
    def length(self):
        return self.segments * 20.

    @property
    def spacing(self):
        return (self.length / (self.c.body_points - 1)).clamp_min(self.c.body_spacing)

    def body_mask(self):
        idx = torch.arange(self.c.body_points - 1, device=self.device)
        return (idx[None, None, :] * self.spacing[..., None] < self.length[..., None]) & self.alive[..., None]

    def active_body_points(self):
        # Exact truncation: the omitted capsules are masked out for every worm.
        count = torch.where(self.alive, (self.length/self.spacing).ceil(), 0)
        return max(1, min(self.c.body_points-1, int(count.max())))

    def _follow(self, new_head):
        path = torch.cat((new_head[:, :, None], self.body), 2)
        segment = path[:, :, 1:] - path[:, :, :-1]
        lengths = segment.norm(dim=-1).clamp_min(1e-6)
        cumulative = torch.cat((torch.zeros_like(lengths[:, :, :1]), lengths.cumsum(-1)), -1)
        targets = (torch.arange(self.c.body_points, device=self.device)[None, None] * self.spacing[..., None]).contiguous()
        idx = torch.searchsorted(cumulative.contiguous(), targets, right=True).clamp(1, self.c.body_points) - 1
        a = path.gather(2, idx[..., None].expand(-1, -1, -1, 2))
        vec = segment.gather(2, idx[..., None].expand(-1, -1, -1, 2))
        t = (targets - cumulative.gather(2, idx)) / lengths.gather(2, idx)
        self.body = a + t[..., None] * vec
        self.head = new_head

    def _collisions(self):
        c = self.c
        count = self.active_body_points()
        a, b = self.body[:, :, :count], self.body[:, :, 1:count+1]
        distances = point_segment_distance2(self.head[:, :, None, None], a[:, None], b[:, None])
        radius = self.radius
        enemy = ~torch.eye(c.worms, device=self.device, dtype=torch.bool)[None, :, :, None]
        hit = (distances < (radius[:, :, None, None] + radius[:, None, :, None]).square()) & self.body_mask()[:, None, :, :count] & enemy
        hit = hit.any(-1)
        collision = hit.any(-1)
        border = self.head.norm(dim=-1) + radius >= self.arena[:, None]
        died = self.alive & (collision | border)
        self.collision_deaths |= died & collision
        self.border_deaths |= died & border & ~collision
        killer = hit.to(torch.int64).argmax(-1)
        self.kills.scatter_add_(1, killer, (died & collision).float())
        # The drop reservoir is separate from ambient food. Its total mass is bounded by the victim's mass.
        mask = self.body_mask()
        count = mask.sum(-1).clamp_min(1)
        drop_mass = torch.where(mask, self.mass[..., None] * .7 / count[..., None], 0.)
        drop_mass = torch.nn.functional.pad(drop_mass, (0, 1))
        reservoir = self.food[:, c.foods:].view(c.maps, c.worms, c.body_points, 2)
        reservoir.copy_(torch.where(died[..., None, None], self.body, reservoir))
        masses = self.food_mass[:, c.foods:].view(c.maps, c.worms, c.body_points)
        masses.copy_(torch.where(died[..., None], drop_mass, masses))
        self.food_size[:, c.foods:] = (self.food_mass[:, c.foods:] * 3).clamp(2, 20)
        self.alive &= ~died
        self.boost &= self.alive

    def _eat(self):
        dist = (self.head[:, :, None] - self.food[:, None]).square().sum(-1)
        can_eat = (dist <= (self.radius[:, :, None] + self.food_size[:, None] * .5).square()) & self.alive[..., None] & (self.food_mass[:, None] > 0)
        closest = torch.where(can_eat, dist, torch.inf)
        best, owner = closest.min(1)
        eaten = torch.isfinite(best)
        gain = torch.zeros_like(self.mass).scatter_add_(1, owner, self.food_mass * eaten)
        self.mass += gain
        self.gained += gain
        ambient = eaten[:, :self.c.foods]
        self.food[:, :self.c.foods] = torch.where(ambient[..., None], self.random_food(self.c.foods), self.food[:, :self.c.foods])
        replacement = .5 + self.rand(self.c.maps, self.c.foods) * 2.5
        self.food_mass[:, :self.c.foods] = torch.where(ambient, replacement, self.food_mass[:, :self.c.foods])
        self.food_mass[:, self.c.foods:] *= ~eaten[:, self.c.foods:]
        self.food_size = (self.food_mass * 3).clamp(2, 20)

    @torch.inference_mode()
    def step(self, actions):
        c = self.c
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).reshape(c.maps, c.worms, 2)
        if not torch.isfinite(actions).all():
            raise ValueError('Non-finite action')
        target = torch.remainder(actions[..., 1], 1.) * math.tau
        delta = target - self.target
        self.turn_sum += torch.atan2(delta.sin(), delta.cos()).abs() * self.alive
        self.decisions += self.alive
        self.target = target
        self.boost = (actions[..., 0] >= .5) & (self.mass > c.min_mass + c.boost_cost * c.dt) & self.alive
        self.boost_steps += self.boost
        cost = self.boost * (c.boost_cost * c.dt)
        self.mass -= cost
        self.spent += cost
        self.speed = torch.where(self.boost, c.boost_speed, c.base_speed) * (1 + (self.radius / 14.5 - 1) * .1)
        # Deposit boost mass at the tail in a finite recycling reservoir.
        p = self.shed_cursor % c.body_points
        ids = c.foods + torch.arange(c.worms, device=self.device) * c.body_points + p
        tail_idx = (self.length / self.spacing).long().clamp(1, c.body_points - 1)
        tail = self.body.gather(2, tail_idx[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        self.food[:, ids] = torch.where(self.boost[..., None], tail, self.food[:, ids])
        self.food_mass[:, ids] = torch.where(self.boost, cost, self.food_mass[:, ids])
        self.food_size[:, ids] = (self.food_mass[:, ids] * 3).clamp(2, 20)
        self.shed_cursor += 1
        if c.preys:
            prey = self.food[:, :c.preys]
            difference = prey[:, :, None] - self.head[:, None]
            dist = torch.where(self.alive[:, None], difference.square().sum(-1), torch.inf)
            nearest, index = dist.min(-1)
            flee = difference.gather(2, index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
            self.prey_heading += (self.rand(c.maps, c.preys) - .5) * .5
            self.prey_heading = torch.where(nearest < 150**2, torch.atan2(flee[..., 1], flee[..., 0]), self.prey_heading)
            self.prey_heading = torch.where(prey.norm(dim=-1) > self.arena[:, None] * .95,
                                            torch.atan2(-prey[..., 1], -prey[..., 0]), self.prey_heading)
            prey += torch.stack((self.prey_heading.cos(), self.prey_heading.sin()), -1) * (75 * c.dt)
        for _ in range(c.substeps):
            delta = torch.atan2((self.target - self.heading).sin(), (self.target - self.heading).cos())
            max_turn = c.turn_rate * (14.5 / self.radius).sqrt() * c.dt / c.substeps
            self.heading += delta.clamp(-max_turn, max_turn) * self.alive
            direction = torch.stack((self.heading.cos(), self.heading.sin()), -1)
            self._follow(self.head + direction * (self.speed * self.alive * c.dt / c.substeps)[..., None])
            self.age += self.alive * c.dt / c.substeps
            self._collisions()
        self._eat()
        self.steps += 1
        self.elapsed += c.dt

    @torch.inference_mode()
    def observe(self):
        c, d = self.c, self.device
        m, w, r = c.maps, c.worms, len(ANGLES)
        n = m * w
        zoom = (self.radius / 14.5).sqrt()
        halfw, halfh = c.view_half_width * zoom, c.view_half_height * zoom
        directions_angle = self.heading[..., None] + self.angles
        direction = torch.stack((directions_angle.cos(), directions_angle.sin()), -1)
        ranges = torch.minimum(halfw[..., None] / direction[..., 0].abs().clamp_min(1e-9), halfh[..., None] / direction[..., 1].abs().clamp_min(1e-9))
        observation = torch.zeros((m, w, r, 6), device=d)
        observation[..., 0] = ranges / (ranges + 1000)
        radius = self.radius
        def prox(hit):
            return torch.where(hit <= ranges, 1 / (1 + hit / 100), 0.)
        # Enemy heads.
        centers = self.head[:, None] - self.head[:, :, None]
        hr = radius[:, :, None] + radius[:, None, :] + 3
        hits = ray_circle(direction[..., None, :], centers[:, :, None], hr[:, :, None])
        mask = self.alive[:, None, :] & ~torch.eye(w, device=d, dtype=torch.bool)[None]
        observation[..., 1] = prox(torch.where(mask[:, :, None], hits, torch.inf).amin(-1))
        # Cull capsules only when their inflated bounding box misses the viewport.
        count = self.active_body_points()
        a = self.body[:, :, :count].reshape(m, -1, 2)
        b = self.body[:, :, 1:count+1].reshape(m, -1, 2)
        owner = torch.arange(w, device=d).repeat_interleave(count)
        mine = owner[None, :] == torch.arange(w, device=d)[:, None]
        ar, br = a[:, None] - self.head[:, :, None], b[:, None] - self.head[:, :, None]
        radii = radius[:, owner][:, None] + torch.where(mine[None], 0., radius[..., None] + 3)
        relevant = (torch.minimum(ar[..., 0], br[..., 0]) - radii <= halfw[..., None]) & (torch.maximum(ar[..., 0], br[..., 0]) + radii >= -halfw[..., None])
        relevant &= (torch.minimum(ar[..., 1], br[..., 1]) - radii <= halfh[..., None]) & (torch.maximum(ar[..., 1], br[..., 1]) + radii >= -halfh[..., None])
        relevant &= self.body_mask()[:, :, :count].reshape(m, 1, -1)
        distance_along = torch.arange(count, device=d).repeat(w)[None, None] * self.spacing[:, owner][:, None]
        relevant &= ~mine[None] | (distance_along >= 3 * radius[..., None])
        max_candidates = int(relevant.sum(-1).max().item())
        if max_candidates:
            # Top-k includes EVERY relevant capsule because k is the maximum relevance count.
            selected = torch.topk(relevant.to(torch.uint8), max_candidates, dim=-1, sorted=False).indices
            valid = relevant.gather(-1, selected).reshape(n, max_candidates)
            own = mine[None].expand(m, -1, -1).gather(-1, selected).reshape(n, max_candidates)
            ga = ar.gather(2, selected[..., None].expand(-1, -1, -1, 2)).reshape(n, max_candidates, 2)
            gb = br.gather(2, selected[..., None].expand(-1, -1, -1, 2)).reshape(n, max_candidates, 2)
            gr = radii.gather(-1, selected).reshape(n, max_candidates)
            flat_dir = direction.reshape(n, r, 2)
            flat_out = observation.reshape(n, r, 6)
            flat_range = ranges.reshape(n, r)
            for start in range(0, n, c.sensor_chunk * w):
                sl = slice(start, start + c.sensor_chunk * w)
                hit = ray_capsule(flat_dir[sl, :, None], ga[sl, None], gb[sl, None], gr[sl, None])
                for channel, matching in ((2, ~own[sl]), (3, own[sl])):
                    near = torch.where((valid[sl] & matching)[:, None], hit, torch.inf).amin(-1)
                    flat_out[sl, :, channel] = torch.where(near <= flat_range[sl], 1 / (1 + near / 100), 0.)
        # Arena: positive ray exit from the collision boundary.
        projection = (self.head[:, :, None] * direction).sum(-1)
        limit = (self.arena[:, None] - radius - 3).clamp_min(0)
        disc = projection.square() + limit[..., None].square() - self.head.square().sum(-1)[..., None]
        border = (-projection + disc.clamp_min(0).sqrt()).clamp_min(0)
        observation[..., 4] = prox(border)
        # Food/proy sector assignment; blind cone is excluded, large bodies do not occlude food.
        relative = self.food[:, None] - self.head[:, :, None]
        theta = torch.atan2(relative[..., 1], relative[..., 0]) - self.heading[..., None]
        theta = torch.atan2(theta.sin(), theta.cos())
        beams = torch.bucketize(theta.contiguous(), self.edges)
        angle = self.heading[..., None] + self.angles[beams]
        projected = relative[..., 0] * angle.cos() + relative[..., 1] * angle.sin()
        entry = (projected - self.food_size[:, None]).clamp_min(0)
        visible = (relative[..., 0].abs() <= halfw[..., None] + self.food_size[:, None]) & (relative[..., 1].abs() <= halfh[..., None] + self.food_size[:, None])
        visible &= (theta.abs() <= math.radians(170)) & (projected >= 0) & (entry <= ranges.gather(-1, beams)) & (self.food_mass[:, None] > 0)
        value = (self.food_size[:, None] / 20).clamp_max(1) / (1 + entry / 100)
        observation[..., 5].scatter_reduce_(-1, beams, torch.where(visible, value, 0.), reduce='amax', include_self=True)
        delta = self.target - self.heading
        globals_ = torch.stack(((self.speed / c.raw_speed_scale / 12).clamp_max(1), (radius / 80).clamp_max(1), (self.segments / 400).clamp_max(1),
                                self.heading.sin(), self.heading.cos(), delta.sin(), delta.cos(), self.boost.float()), -1)
        result = torch.cat((observation.flatten(2), globals_), -1).reshape(n, INPUTS)
        return result * self.alive.reshape(n, 1)

    def fitness(self):
        return reward_score(self.gained, self.spent, self.age, self.kills, self.alive, self.c.reward_version)

    def fitness_terms(self):
        return reward_terms(self.gained, self.spent, self.age, self.kills, self.alive, self.c.reward_version)

    def snapshot(self, arena=0):
        arena = int(arena) % self.c.maps
        mask = self.body_mask()[arena].sum(-1).cpu().tolist()
        body = self.body[arena].cpu().tolist()
        mass = self.mass[arena].cpu().tolist()
        alive = self.alive[arena].cpu().tolist()
        food_alive = self.food_mass[arena] > 0
        return dict(arena=arena, radius=float(self.arena[arena]), elapsed=round(self.elapsed, 2), config=asdict(self.c),
                    worms=[dict(id=i, alive=alive[i], mass=round(mass[i], 1), boost=bool(self.boost[arena, i]),
                                body=[[round(x, 1), round(y, 1)] for x, y in body[i][:mask[i]+1]], radius=round(float(self.radius[arena, i]), 1)) for i in range(self.c.worms)],
                    food=torch.cat((self.food[arena, food_alive], self.food_size[arena, food_alive, None]), -1).round(decimals=1).cpu().tolist(),
                    maps=[dict(id=i, alive=int(n), mean_mass=round(float(s), 1)) for i, (n, s) in enumerate(zip(self.alive.sum(1).cpu(), self.mass.mean(1).cpu()))])
