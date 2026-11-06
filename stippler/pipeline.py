"""The three pipeline stages tied together, plus settings validation.

make_density(a) -> prep      image -> density map           (stage 1)
make_points(a, prep) -> pts  seed + relax the dots          (stage 2)
render(a, prep, pts, out)    jitter, size, lines, write     (stage 3)

`a` is the argparse namespace from cli.build_parser().
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .density import load_density, load_grey
from .drawing import clamp_to_neighbours, dot_radii, render_png, render_svg
from .relax import initial_points, jitter_points, lloyd_relax


# Settings that each pipeline stage depends on (used by sweep.py for caching).
# Everything else (dot size, jitter, lines, output size) is applied at render
# time, so sweeping those options reuses the relaxed dots.
DENSITY_KEYS = ("gamma", "contrast", "blur", "invert", "sharpen", "white_cutoff",
                "simplify", "simplify_passes", "levels", "points", "work_size",
                "focus", "focus_local")
POINTS_KEYS = DENSITY_KEYS + ("iterations", "seed", "lines_only")


def make_density(a):
    """Stage 1: image -> density map."""
    density, orig_size = load_density(a.input, a.work_size, a.gamma, a.contrast,
                                      a.blur, a.invert, a.sharpen, a.white_cutoff,
                                      a.simplify, a.simplify_passes, a.levels,
                                      a.points, a.focus, a.focus_local)
    return SimpleNamespace(density=density, orig_size=orig_size, line_cache={})


def make_points(a, prep):
    """Stage 2: place and relax the dots (lines play no part in this)."""
    if a.lines_only or a.points <= 0:
        return np.empty((0, 2))
    rng = np.random.default_rng(a.seed)
    pts = initial_points(prep.density, a.points, rng)
    if a.iterations > 0:
        pts = lloyd_relax(pts, prep.density, a.iterations, rng, verbose=not a.quiet)
    return pts


def get_lines(a, prep):
    """Detect outlines (cached per detail/min-length so sweeps reuse them)."""
    key = (a.line_detail, a.min_line_length)
    if key not in prep.line_cache:
        from .lines import detect_lines
        prep.line_cache[key] = detect_lines(load_grey(a.input), *key)
    return prep.line_cache[key]


def render(a, prep, pts, out):
    """Stage 3: jitter, size the dots, add lines, write PNG (and SVG)."""
    density = prep.density
    h, w = density.shape
    out_w = a.out_width or round(prep.orig_size[0] * a.scale)
    scale = out_w / w
    out_size = (out_w, max(1, round(h * scale)))

    pts = jitter_points(pts, a.jitter, a.seed, density.shape)

    radius = 0.0
    radii = np.empty(0)
    if len(pts):
        if a.dot_radius is None:
            # Auto: size dots so the darkest areas come out close to solid black.
            # Average spacing in fully black area ~ sqrt(ink_area / n).
            ink_area = density.sum() * scale * scale
            spacing = np.sqrt(ink_area / len(pts))
            radius = max(0.6, 0.35 * spacing)
        else:
            radius = a.dot_radius
        radii = dot_radii(pts, density, radius, a.variable_size)
        if not a.allow_touching:
            radii = clamp_to_neighbours(pts, radii, scale)

    lines, line_width = [], 0.0
    if a.lines or a.lines_only:
        lines = get_lines(a, prep)
        if a.line_width:
            line_width = a.line_width
        elif len(pts):
            line_width = max(1.0, 0.25 * np.sqrt(out_size[0] * out_size[1] / len(pts)))
        else:
            line_width = max(1.5, np.hypot(*out_size) / 1500)
        if a.line_gap is not None and len(pts) and lines:
            # Optional: drop dots that would touch / crowd a line.
            from .lines import line_distance_map
            dist = line_distance_map(lines, out_size, line_width)  # px from line edge
            spacing_out = np.sqrt(out_size[0] * out_size[1] / len(pts))
            xi = np.clip((pts[:, 0] * scale).astype(int), 0, out_size[0] - 1)
            yi = np.clip((pts[:, 1] * scale).astype(int), 0, out_size[1] - 1)
            keep = dist[yi, xi] > radii + a.line_gap * spacing_out
            pts, radii = pts[keep], radii[keep]

    out = Path(out)
    render_png(pts, radii, scale, out_size, out, lines, line_width)
    if not a.quiet:
        msg = f"Wrote {out}  ({out_size[0]}x{out_size[1]}, {len(pts)} dots r={radius:.2f}px"
        if lines:
            msg += f", {len(lines)} lines w={line_width:.2f}px"
        print(msg + ")", file=sys.stderr)
    if a.svg:
        render_svg(pts, radii, scale, out_size, out.with_suffix(".svg"), lines, line_width)
    return out


def validate(a):
    """Return a list of problems with the settings (empty if all fine)."""
    errs = []

    def need(cond, msg):
        if not cond:
            errs.append(msg)

    need(a.scale > 0, f"--scale must be > 0 (got {a.scale:g}; 1 = input size)")
    need(a.out_width is None or a.out_width > 0, "--out-width must be > 0")
    need(a.points >= 1 or a.lines_only, "--points must be >= 1")
    need(a.iterations >= 0, "--iterations must be >= 0")
    need(a.gamma > 0, "--gamma must be > 0")
    need(a.contrast >= 0, "--contrast must be >= 0")
    need(0 <= a.white_cutoff < 1, "--white-cutoff must be in [0, 1)")
    need(a.blur >= 0, "--blur must be >= 0")
    need(a.simplify >= 0, "--simplify must be >= 0")
    need(a.simplify_passes >= 1, "--simplify-passes must be >= 1")
    need(a.levels >= 0 and a.levels != 1, "--levels must be 0 (off) or >= 2")
    need(a.sharpen >= 0, "--sharpen must be >= 0")
    need(a.work_size >= 0, "--work-size must be >= 0 (0 = auto)")
    need(a.jitter >= 0, "--jitter must be >= 0")
    need(a.dot_radius is None or a.dot_radius > 0, "--dot-radius must be > 0")
    need(a.line_detail > 0, "--line-detail must be > 0")
    need(a.min_line_length >= 0, "--min-line-length must be >= 0")
    need(a.line_width is None or a.line_width > 0, "--line-width must be > 0")
    need(a.line_gap is None or a.line_gap >= 0, "--line-gap must be >= 0")
    if a.focus and a.focus.strip().lower() != "select":
        from .focus import parse_spec
        try:
            parse_spec(a.focus)
        except ValueError as e:
            errs.append(str(e))
    return errs
