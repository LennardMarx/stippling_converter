"""Focus region: optimise the tone mapping for one part of the image.

A region is given as
  - a box  "x0,y0,x1,y1"  as fractions of the image (0-1) or in pixels,
  - a mask image path      (white = region, any size; resized to fit),
  - "select"               to drag a box on the image interactively.

The black point, white point and mid-tone are then measured on that region
only, so its tones spread over the full range of dot densities. In local mode
this mapping is blended in only around the region (soft edge) and the rest of
the image keeps the normal global contrast.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BOX_RE = re.compile(r"^\s*-?[\d.]+(\s*[,:]\s*-?[\d.]+){3}\s*$")


def parse_spec(spec):
    """Return ('box', (x0, y0, x1, y1)) or ('mask', Path). Raises ValueError."""
    if spec is None:
        return None
    if BOX_RE.match(spec):
        x0, y0, x1, y1 = (float(v) for v in re.split(r"[,:]", spec))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"--focus box needs x0 < x1 and y0 < y1 (got {spec})")
        return "box", (x0, y0, x1, y1)
    if spec.strip().lower() == "select":
        raise ValueError("--focus select must be resolved before processing")
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"--focus: '{spec}' is neither a box x0,y0,x1,y1 nor a mask file")
    return "mask", path


def region_mask(spec, work_shape, orig_size):
    """Focus region as a float mask in [0, 1] at the work resolution."""
    kind, val = parse_spec(spec)
    h, w = work_shape
    if kind == "mask":
        m = Image.open(val).convert("L").resize((w, h), Image.BILINEAR)
        return np.asarray(m, dtype=np.float64) / 255.0
    x0, y0, x1, y1 = val
    if max(val) > 1.0:  # pixel coordinates of the original image
        ow, oh = orig_size
        x0, x1, y0, y1 = x0 / ow, x1 / ow, y0 / oh, y1 / oh
    mask = np.zeros((h, w))
    c0, c1 = int(np.clip(x0, 0, 1) * w), int(np.ceil(np.clip(x1, 0, 1) * w))
    r0, r1 = int(np.clip(y0, 0, 1) * h), int(np.ceil(np.clip(y1, 0, 1) * h))
    mask[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = 1.0
    return mask


def metered_levels(lum, select):
    """Stretch so the selected pixels span black..white, with their median
    tone mapped to mid-grey."""
    vals = lum[select] if select is not None else lum.ravel()
    if vals.size < 16:
        vals = lum.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    hi = max(hi, lo + 1e-3)
    x = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)
    med = float(np.clip(np.median(np.clip((vals - lo) / (hi - lo), 0, 1)), 0.05, 0.95))
    return x ** (np.log(0.5) / np.log(med))


def select_region(path, max_view=(1400, 900)):
    """Let the user drag a box on the image. Returns 'x0,y0,x1,y1' fractions."""
    import cv2
    from .density import load_grey

    img = np.asarray(load_grey(path))
    h, w = img.shape
    s = min(1.0, max_view[0] / w, max_view[1] / h)
    view = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    title = "Drag a box over the focus region, then press ENTER (C = cancel)"
    x, y, bw, bh = cv2.selectROI(title, view, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if bw == 0 or bh == 0:
        raise SystemExit("No focus region selected.")
    vw, vh = view.shape[1], view.shape[0]
    return f"{x / vw:.3f},{y / vh:.3f},{(x + bw) / vw:.3f},{(y + bh) / vh:.3f}"


def feather(mask, amount=0.06):
    """Mask that stays 1 inside the region and fades out smoothly around it;
    amount is the fade width as a fraction of the image diagonal."""
    sigma = amount * np.hypot(*mask.shape)
    soft = gaussian_filter(mask, sigma)
    peak = soft.max()
    if peak > 0:
        soft = soft / peak
    return np.clip(np.maximum(mask, soft), 0.0, 1.0)
