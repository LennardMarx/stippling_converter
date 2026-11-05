"""Convert an image into a stippling-style drawing (black dots on white).

Method: weighted Voronoi stippling (Secord 2002).
  1. Convert to greyscale; darkness becomes a density map.
  2. Scatter dots randomly, more where the image is darker.
  3. Lloyd relaxation: move every dot to the density-weighted centroid of
     its Voronoi cell. This spreads dots evenly while keeping the tone,
     giving the hand-stippled look instead of random noise.
  4. Render the dots as pure black circles on white (binary output).

Line mode (--lines): the most important edges are traced as thin vector
outlines (see lines.py) and drawn on top of the stippling. They don't affect
where dots go (use --line-gap to clear dots off the lines if wanted).

Example:
    python stipple.py photo.jpg -n 12000 -o photo_stipple.png --svg
"""

__all__ = ["cli", "sweep", "pipeline"]


