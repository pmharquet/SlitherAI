"""Broadcastable analytic intersections, in world units."""
import torch

def point_segment_distance2(p, a, b):
    ab = b - a
    t = ((p - a) * ab).sum(-1) / ab.square().sum(-1).clamp_min(1e-12)
    closest = a + t.clamp(0, 1).unsqueeze(-1) * ab
    return (p - closest).square().sum(-1)

def ray_circle(direction, center, radius):
    """Ray origin is (0,0); no hit is +inf."""
    projection = (center * direction).sum(-1)
    disc = radius.square() - center.square().sum(-1) + projection.square()
    root = disc.clamp_min(0).sqrt()
    return torch.where((disc >= 0) & (projection + root >= 0), (projection - root).clamp_min(0), torch.inf)

def ray_capsule(direction, a, b, radius):
    ab = b - a
    length = ab.square().sum(-1).sqrt()
    u = ab / length.clamp_min(1e-9).unsqueeze(-1)
    normal = torch.stack((-u[..., 1], u[..., 0]), -1)
    def slab(position, velocity, low, high):
        parallel = velocity.abs() < 1e-8
        safe = torch.where(parallel, torch.ones_like(velocity), velocity)
        x, y = (low - position) / safe, (high - position) / safe
        inside = (position >= low) & (position <= high)
        lower = torch.where(parallel, torch.where(inside, -torch.inf, torch.inf), torch.minimum(x, y))
        upper = torch.where(parallel, torch.where(inside, torch.inf, -torch.inf), torch.maximum(x, y))
        return lower, upper
    lo1, hi1 = slab((-a * u).sum(-1), (direction * u).sum(-1), 0., length)
    lo2, hi2 = slab((-a * normal).sum(-1), (direction * normal).sum(-1), -radius, radius)
    entry, exit = torch.maximum(lo1, lo2).clamp_min(0), torch.minimum(hi1, hi2)
    strip = torch.where((exit >= entry) & (length > 1e-8), entry, torch.inf)
    return torch.minimum(strip, torch.minimum(ray_circle(direction, a, radius), ray_circle(direction, b, radius)))
