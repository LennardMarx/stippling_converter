"""Stage 1: image -> greyscale -> preprocessing -> dot density map."""

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import uniform_filter


def kuwahara(img, radius):
    """Kuwahara filter: edge-preserving smoothing that flattens texture into
    uniform patches while keeping outlines sharp. img is float [0,1]."""
    r = max(1, int(radius))
    k = r + 1                                   # quadrant window size
    mean = uniform_filter(img, k, mode="reflect")
    var = uniform_filter(img * img, k, mode="reflect") - mean * mean
    # The window centred at (y, x) is a quadrant for pixels offset by +-r/2.
    lo, hi = r // 2, r - r // 2
    pad = r
    mp = np.pad(mean, pad, mode="edge")
    vp = np.pad(var, pad, mode="edge")
    h, w = img.shape
    means, vars_ = [], []
    for dy in (-lo, hi):
        for dx in (-lo, hi):
            sl = (slice(pad + dy, pad + dy + h), slice(pad + dx, pad + dx + w))
            means.append(mp[sl])
            vars_.append(vp[sl])
    means = np.stack(means)
    best = np.argmin(np.stack(vars_), axis=0)
    return np.take_along_axis(means, best[None], axis=0)[0]


def simplify(lum, radius, passes, levels):
    """Flatten fine texture (windows, foliage, noise) smaller than the dot
    spacing so it doesn't turn into random dot clumps."""
    for _ in range(passes):
        lum = kuwahara(lum, radius)
    if levels and levels > 1:
        lum = np.round(lum * (levels - 1)) / (levels - 1)
    return lum


def load_grey(path):
    """Open an image (EXIF-rotated, transparency on white) as greyscale."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def load_density(path, work_size, gamma, contrast, blur, invert, sharpen=0.0,
                 white_cutoff=0.0, simplify_radius=0.0, simplify_passes=2,
                 levels=0, n_points=None):
    """Load image, convert to greyscale and return a density map in [0, 1]."""
    grey = load_grey(path)
    orig_size = grey.size

    if work_size:
        scale = min(1.0, work_size / max(grey.size))
    else:
        # Auto: give every dot ~150 pixels of density map so Lloyd's centroids
        # are precise; too coarse a grid leaves dots stuck in clumps and chains.
        target = (n_points or 50000) * 150
        scale = min(1.0, max(800 / max(grey.size), np.sqrt(target / (grey.width * grey.height))))
    if scale < 1.0:
        new = (max(1, round(grey.width * scale)), max(1, round(grey.height * scale)))
        grey = grey.resize(new, Image.LANCZOS)
    if blur > 0:
        grey = grey.filter(ImageFilter.GaussianBlur(blur))
    if simplify_radius > 0 or levels:
        # Radius is relative to the typical dot spacing in the work image.
        spacing = np.sqrt(grey.width * grey.height / n_points) if n_points else 4
        r = max(1, round(simplify_radius * spacing)) if simplify_radius > 0 else 0
        lum = np.asarray(grey, dtype=np.float64) / 255.0
        lum = simplify(lum, r, simplify_passes if r else 0, levels)
        grey = Image.fromarray(np.clip(lum * 255 + 0.5, 0, 255).astype(np.uint8))
    if sharpen > 0:
        # Unsharp mask boosts edges/local contrast so outlines survive as dots.
        grey = grey.filter(ImageFilter.UnsharpMask(
            radius=max(2, grey.width // 200), percent=int(sharpen * 100), threshold=2))
    if contrast > 0:
        grey = ImageOps.autocontrast(grey, cutoff=1)

    lum = np.asarray(grey, dtype=np.float64) / 255.0
    if contrast > 0:
        lum = s_curve(lum, contrast)
    density = lum if invert else 1.0 - lum
    density = np.clip(density, 0.0, 1.0) ** gamma
    if white_cutoff > 0:
        # Light tones below the cutoff become pure white paper; rescale the rest.
        density = np.clip((density - white_cutoff) / (1.0 - white_cutoff), 0.0, 1.0)
    return density, orig_size


def s_curve(x, strength):
    """Sigmoid contrast curve on [0,1]: darkens darks, lightens lights."""
    k = 1.0 + 4.0 * strength
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    lo, hi = sig(-k / 2), sig(k / 2)
    return (sig(k * (x - 0.5)) - lo) / (hi - lo)
