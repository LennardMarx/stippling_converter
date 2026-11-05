"""Command-line options and the single-image entry point."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import pipeline


def build_parser():
    p = argparse.ArgumentParser(
        prog="stipple.py",
        description="Convert an image into a binary stippling drawing. "
                    "Use 'stipple.py sweep ...' to try many settings at once.")
    p.add_argument("input", help="input image")
    p.add_argument("-o", "--output", help="output PNG (default: <input>_stipple.png)")
    p.add_argument("-n", "--points", type=int, default=50000,
                   help="number of dots (default 50000)")
    p.add_argument("-i", "--iterations", type=int, default=60,
                   help="Lloyd relaxation iterations (default 60; 0 = random)")
    p.add_argument("-r", "--dot-radius", type=float, default=None,
                   help="dot radius in output pixels (default: auto)")
    p.add_argument("--variable-size", action="store_true",
                   help="make dots bigger in darker areas")
    p.add_argument("--gamma", type=float, default=1.0,
                   help=">1 lightens midtones (fewer dots), <1 darkens")
    p.add_argument("-c", "--contrast", type=float, default=1.5,
                   help="contrast boost: autocontrast + S-curve (default 1.5, 0 = off)")
    p.add_argument("-w", "--white-cutoff", type=float, default=0.12,
                   help="tones lighter than this (0-1) get no dots (default 0.12)")
    p.add_argument("--blur", type=float, default=0.0,
                   help="Gaussian blur radius on the working image (smooths noise)")
    p.add_argument("--simplify", type=float, default=0.0,
                   help="flatten texture finer than the dot spacing, keeping edges "
                        "(Kuwahara filter; radius x dot spacing, e.g. 0.7; default 0 = off)")
    p.add_argument("--simplify-passes", type=int, default=2,
                   help="Kuwahara passes; more = flatter shapes (default 2)")
    p.add_argument("--levels", type=int, default=0,
                   help="posterize to N flat grey levels after simplifying (e.g. 5; 0 = off)")
    p.add_argument("--sharpen", type=float, default=1.5,
                   help="edge enhancement strength (default 1.5, 0 = off)")
    p.add_argument("--invert", action="store_true",
                   help="dots for bright areas instead of dark")
    p.add_argument("--work-size", type=int, default=0,
                   help="longest side of the internal density map "
                        "(default 0 = auto, ~150 px per dot)")
    p.add_argument("--allow-touching", action="store_true",
                   help="don't shrink dots that are close to a neighbour")
    p.add_argument("-s", "--scale", type=float, default=2.0,
                   help="output size relative to input (default 2 -> finer, rounder dots)")
    p.add_argument("--out-width", type=int, default=None,
                   help="output width in pixels (overrides --scale)")
    g = p.add_argument_group("line mode")
    g.add_argument("--lines", action="store_true",
                   help="draw thin outlines on the most important edges, stipple inside")
    g.add_argument("--lines-only", action="store_true",
                   help="only the outlines, no dots")
    g.add_argument("--line-detail", type=float, default=1.0,
                   help="higher = more and finer lines (default 1; try 0.5-2)")
    g.add_argument("--min-line-length", type=float, default=0.03,
                   help="drop lines shorter than this fraction of the image "
                        "diagonal (default 0.03)")
    g.add_argument("--line-width", type=float, default=None,
                   help="line width in output pixels (default: auto, thinner than dots)")
    g.add_argument("--line-gap", type=float, default=None,
                   help="optional: remove dots that come within this margin of a "
                        "line, in dot spacings (0 = only dots touching a line). "
                        "Default: off, lines are simply drawn on top of the dots")
    p.add_argument("-j", "--jitter", type=float, default=0.25,
                   help="randomly offset each dot by up to this fraction of the "
                        "distance to its neighbour; breaks up the regular "
                        "star/crystal patterns (0 = off, 0.5 = quite random; default 0.25)")
    p.add_argument("--svg", action="store_true", help="also write an SVG file")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None):
    """Stipple a single image."""
    parser = build_parser()
    a = parser.parse_args(argv)
    errs = pipeline.validate(a)
    if errs:
        parser.error("; ".join(errs))
    Image.MAX_IMAGE_PIXELS = None  # allow large photos / outputs
    prep = pipeline.make_density(a)
    if not a.quiet:
        h, w = prep.density.shape
        print(f"Density map {w}x{h}, {0 if a.lines_only else a.points} dots",
              file=sys.stderr)
    pts = pipeline.make_points(a, prep)
    out = a.output or Path(a.input).with_name(Path(a.input).stem + "_stipple.png")
    pipeline.render(a, prep, pts, out)
