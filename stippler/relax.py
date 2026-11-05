"""Stage 2: dot placement - random seeding, weighted Lloyd relaxation, jitter."""

import sys

import numpy as np
from scipy.spatial import cKDTree


def initial_points(density, n, rng):
    """Rejection-sample n points with probability proportional to density."""
    h, w = density.shape
    flat = density.ravel()
    total = flat.sum()
    if total <= 0:
        raise ValueError("Image is completely white - nothing to stipple.")
    idx = rng.choice(flat.size, size=n, p=flat / total)
    ys, xs = np.divmod(idx, w)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts += rng.random(pts.shape)  # jitter inside the pixel
    return pts


def lloyd_relax(points, density, iterations, rng, verbose=True):
    """Weighted Lloyd relaxation of points over the density map."""
    h, w = density.shape
    mask = density > 1e-4
    ys, xs = np.nonzero(mask)
    wts = density[mask]
    px = xs + 0.5
    py = ys + 0.5
    coords = np.column_stack([px, py])
    n = len(points)

    for it in range(iterations):
        _, owner = cKDTree(points).query(coords, workers=-1)
        mass = np.bincount(owner, weights=wts, minlength=n)
        cx = np.bincount(owner, weights=wts * px, minlength=n)
        cy = np.bincount(owner, weights=wts * py, minlength=n)

        has = mass > 0
        new = points.copy()
        new[has, 0] = cx[has] / mass[has]
        new[has, 1] = cy[has] / mass[has]

        # Dots that ended up owning no ink get re-seeded somewhere dark.
        empty = np.count_nonzero(~has)
        if empty:
            new[~has] = initial_points(density, empty, rng)

        shift = np.sqrt(((new - points) ** 2).sum(axis=1)).mean()
        points = new
        if verbose:
            print(f"  iteration {it + 1:3d}/{iterations}  mean shift {shift:.4f} px",
                  file=sys.stderr)
        if shift < 0.002:
            break
    return points


def jitter_points(pts, amount, seed, shape):
    """Offset each dot randomly (uniform in a disc) by up to `amount` times the
    distance to its nearest neighbour. Relaxed dots settle into hexagonal
    patches whose seams read as stars; a little disorder hides them."""
    if amount <= 0 or len(pts) < 2:
        return pts
    rng = np.random.default_rng(None if seed is None else seed + 7919)
    nn = cKDTree(pts).query(pts, k=2, workers=-1)[0][:, 1]
    r = amount * nn * np.sqrt(rng.random(len(pts)))
    t = rng.random(len(pts)) * 2 * np.pi
    out = pts + np.column_stack([r * np.cos(t), r * np.sin(t)])
    h, w = shape
    out[:, 0] = np.clip(out[:, 0], 0, w)
    out[:, 1] = np.clip(out[:, 1], 0, h)
    return out
