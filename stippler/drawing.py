"""Stage 3: dot sizing and output (1-bit PNG / SVG)."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


def dot_radii(points, density, base_radius, variable):
    """Per-dot radius; optionally larger dots in darker regions."""
    if not variable:
        return np.full(len(points), base_radius)
    h, w = density.shape
    xi = np.clip(points[:, 0].astype(int), 0, w - 1)
    yi = np.clip(points[:, 1].astype(int), 0, h - 1)
    d = density[yi, xi]
    return base_radius * (0.5 + 0.75 * np.sqrt(d))


def clamp_to_neighbours(points, radii, scale, max_fill=0.42):
    """Shrink dots that sit too close to a neighbour, so dots never touch or
    merge into blobs. max_fill = max radius as a fraction of the NN distance."""
    pts = points * scale
    dist, _ = cKDTree(pts).query(pts, k=2, workers=-1)
    return np.minimum(radii, np.maximum(0.5, max_fill * dist[:, 1]))


def render_png(points, radii, scale, out_size, path, lines=(), line_width=1.0):
    """Draw pure black lines and dots on white, 1-bit image."""
    img = Image.new("1", out_size, 1)
    draw = ImageDraw.Draw(img)
    lw = max(1, int(round(line_width)))
    for line, closed in lines:
        xy = [tuple(p) for p in line * out_size]
        if closed:
            xy = xy + xy[:1]
        draw.line(xy, fill=0, width=lw, joint="curve" if lw > 2 else None)
    for (x, y), r in zip(points * scale, radii):
        if r <= 0.5:
            draw.point((x, y), fill=0)
        else:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
    img.save(path)


def render_svg(points, radii, scale, out_size, path, polylines=(), line_width=1.0):
    w, h = out_size
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        '<g fill="black">',
    ]
    for (x, y), r in zip(points * scale, radii):
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}"/>')
    lines.append("</g>")
    if len(polylines):
        lines.append(f'<g fill="none" stroke="black" stroke-width="{line_width:.2f}" '
                     'stroke-linejoin="round" stroke-linecap="round">')
        for pl, closed in polylines:
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pl * out_size)
            lines.append(f'<path d="M{d}{"Z" if closed else ""}"/>')
        lines.append("</g>")
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
