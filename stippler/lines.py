"""Outline extraction for line mode: find the most important edges of an image
and return them as smooth vector polylines.

Pipeline: greyscale -> autocontrast -> edge-preserving smoothing (bilateral)
-> Gaussian blur (removes fine texture) -> Canny with thresholds taken from
the gradient statistics (only the strongest edges) -> walk the 1-px edge
pixels into single-stroke polylines -> drop short ones -> simplify + smooth.
"""

import cv2
import numpy as np
from PIL import Image, ImageOps

ANALYSIS_SIZE = 1200  # long side used for edge detection (scale-independent results)

# 4-neighbours first so walks follow straight runs before cutting corners.
NEIGHBOURS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def chaikin(pts, closed, iterations=2):
    """Corner-cutting smoothing; open lines keep their end points."""
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        nxt = np.roll(pts, -1, axis=0) if closed else pts[1:]
        cur = pts if closed else pts[:-1]
        q = 0.75 * cur + 0.25 * nxt
        r = 0.25 * cur + 0.75 * nxt
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = mid if closed else np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def trace_edges(edges):
    """Turn a 1-px edge map into polylines by walking neighbouring pixels.
    Returns a list of (points[k, 2] as x, y; closed: bool)."""
    mask = edges > 0
    h, w = mask.shape
    # Neighbour count per pixel -> end points (1 neighbour) start the walks.
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    deg = cv2.filter2D(mask.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero(mask)
    order = np.argsort(deg[ys, xs] != 1, kind="stable")  # end points first
    visited = np.zeros_like(mask)
    paths = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if visited[y, x]:
            continue
        start = (y, x)
        visited[y, x] = True
        path = [(x, y)]
        while True:
            for dy, dx in NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    y, x = ny, nx
                    path.append((x, y))
                    break
            else:
                break
        closed = (len(path) > 8 and abs(y - start[0]) <= 1 and abs(x - start[1]) <= 1)
        paths.append((np.array(path, dtype=np.float64), closed))
    return paths


def detect_lines(grey, detail=1.0, min_length=0.03):
    """Return a list of (polyline, closed) with polylines as N x 2 arrays in
    normalised coordinates (x / width, y / height).

    detail:      higher = more and finer lines (default 1)
    min_length:  drop lines shorter than this fraction of the image diagonal
    """
    scale = min(1.0, ANALYSIS_SIZE / max(grey.size))
    if scale < 1.0:
        grey = grey.resize((round(grey.width * scale), round(grey.height * scale)),
                           Image.LANCZOS)
    grey = ImageOps.autocontrast(grey, cutoff=1)
    g = np.asarray(grey, dtype=np.uint8)
    h, w = g.shape

    # Flatten texture but keep real boundaries.
    for _ in range(2):
        g = cv2.bilateralFilter(g, 9, 40, 7)
    sigma = 1.8 / max(detail, 0.1)
    g = cv2.GaussianBlur(g, (0, 0), sigma)

    # Thresholds from the gradient distribution: only the top few percent of
    # edge strength start a line (more with higher detail).
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    pct = float(np.clip(100 - 6 * detail, 70, 99.5))
    high = max(float(np.percentile(mag, pct)), 1.0)
    edges = cv2.Canny(g, 0.4 * high, high, L2gradient=True)

    diag = np.hypot(w, h)
    out = []
    for path, closed in trace_edges(edges):
        if len(path) < min_length * diag:  # ~1 px per step
            continue
        c = cv2.approxPolyDP(path.reshape(-1, 1, 2).astype(np.float32), 0.8, closed)
        c = c.reshape(-1, 2).astype(np.float64) + 0.5
        if len(c) < 2:
            continue
        c = chaikin(c, closed, 2)
        c[:, 0] /= w
        c[:, 1] /= h
        out.append((c, closed))
    return out


def line_distance_map(lines, size, width):
    """Distance (pixels) from each pixel of a size=(w, h) grid to the nearest
    line, with lines drawn `width` pixels wide."""
    w, h = size
    canvas = np.full((h, w), 255, np.uint8)
    for line, closed in lines:
        pts = np.round(line * (w, h)).astype(np.int32)
        cv2.polylines(canvas, [pts], closed, 0, thickness=max(1, int(round(width))))
    return cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
