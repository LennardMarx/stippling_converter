# Stippling Converter

Turns an image into a binary stippling drawing (black dots on white) using
weighted Voronoi stippling: greyscale -> darkness becomes dot density ->
dots are relaxed (Lloyd's algorithm) so they're evenly spaced like hand stippling.

```
pip install -r requirements.txt
python stipple.py photo.jpg                      # -> photo_stipple.png
python stipple.py photo.jpg -n 100000 -s 3 --svg  # even finer, also SVG
python stipple.py photo.jpg -c 0 -w 0 --sharpen 0 # plain, no tone boosting
```

## Options

| Option | Effect |
|---|---|
| `-n / --points` | number of dots (default 50000) — main detail control |
| `-i / --iterations` | relaxation steps (default 60; 0 = random scatter) |
| `-r / --dot-radius` | dot radius in output px (default: auto) |
| `-j / --jitter` | random offset per dot (fraction of neighbour distance) to break up star/crystal patterns (default 0.25; 0 = off, 0.5 = quite random) |
| `--variable-size` | bigger dots in darker areas |
| `--gamma` | >1 lightens midtones, <1 darkens |
| `-c / --contrast` | autocontrast + S-curve strength (default 1.5, 0 = off) |
| `-w / --white-cutoff` | tones lighter than this get no dots (default 0.12) |
| `--blur` | smooth noisy photos before stippling |
| `--simplify` | flatten texture finer than the dot spacing into clean shapes, keeping edges (Kuwahara filter; try 0.5–1; default off) |
| `--simplify-passes` | more passes = flatter shapes (default 2) |
| `--levels` | posterize to N flat tones, e.g. 4–6 (default off) |
| `--sharpen` | edge enhancement (default 1.5, 0 = off) |
| `-s / --scale` | output size relative to input (default 2) |
| `--out-width` | output width in px (overrides `--scale`) |
| `--work-size` | internal resolution (default auto, ~150 px per dot — lower values cause dot clumps) |
| `--allow-touching` | don't shrink dots that crowd a neighbour |
| `--svg` | also write a scalable SVG |
| `--seed` | reproducible results |

## Parameter sweep

`stipple.py sweep` renders one image with **every combination** of settings into a
folder under `sweeps/`, plus `_contact_sheet.png` (all results side by side, labelled) and
`settings.csv`. Any normal option can be a comma-separated list:

```
python stipple.py sweep skyline.jpg -n 20000,50000 -c 1,2.5 --simplify 0,0.7
# -> sweeps/skyline_sweep/ with 8 images
python stipple.py sweep flower.avif --variable-size no,yes -r auto,2 --dry-run   # just list combos
```

- On/off flags take `no,yes`; `-r auto` means the automatic dot size.
- Unlisted options use the normal defaults, except output scale = 1 and seed = 1
  (so results only differ by the swept settings). Override with e.g. `-s 2`.
- Runs that share a density map or dot layout reuse it, so sweeping render-only
  options (`-r`, `-j`, `--variable-size`, `-s`, all line options) is nearly free.
- Combinations multiply quickly — use `--dry-run` to check the count first.

## Line mode

`--lines` traces the most important edges as thin outlines and draws them on
top of the stippling. The lines don't change where the dots go; add
`--line-gap` if you want dots cleared off the lines.

```
python stipple.py flower.avif --lines
python stipple.py skyline.jpg --lines --line-detail 0.6     # only main silhouettes
python stipple.py skyline.jpg --lines-only --svg            # just the outline drawing
python stipple.py sweep flower.avif --lines yes --line-detail 0.6,1,1.6
python stipple.py sweep flower.avif --lines yes --line-gap off,0,0.3
```

| Option | Effect |
|---|---|
| `--lines` | outlines + stippling |
| `--lines-only` | outlines only, no dots |
| `--line-detail` | higher = more/finer lines (default 1; try 0.5–2) |
| `--min-line-length` | drop lines shorter than this fraction of the image diagonal (default 0.03) |
| `--line-width` | line width in output px (default auto, thinner than the dots) |
| `--line-gap` | optional: remove dots within this margin of a line, in dot spacings (0 = only dots touching a line; default off) |

Edges are found with OpenCV (bilateral smoothing → Canny with adaptive
thresholds) and traced into single-stroke smoothed polylines, which are also
written to the SVG as real paths.

## Focus region

Optimise the contrast for one part of the image: its black point, white point
and mid-tone are measured on that region only, so its tones use the full range
of dot densities.

```
python stipple.py photo.jpg --focus select                 # drag a box on the image
python stipple.py photo.jpg --focus 0.3,0.2,0.6,0.7        # box x0,y0,x1,y1 (fractions or pixels)
python stipple.py photo.jpg --focus mask.png               # painted mask, white = region
python stipple.py photo.jpg --focus select --focus-local   # only boost the region
```

- `--focus select` prints the box it used, so you can repeat it with the same values.
- Default: the whole image gets the region-optimised tone curve (other areas
  may go very light or very dark). `--focus-local` applies it only in and
  softly around the region; the rest keeps the normal contrast.
- Combines with `-c` (S-curve strength on top of the region-metered levels).
- In sweeps, separate several focus values with `;` (boxes contain commas):
  `--focus "none;0.3,0.2,0.6,0.7;mask.png" --focus-local no,yes`.
  `select` is asked once, before the sweep starts.

## Project layout

```
stipple.py            entry point: `stipple.py <image>` or `stipple.py sweep <image>`
requirements.txt
stippler/
  cli.py              command-line options, single-image run
  sweep.py            parameter sweep + contact sheet
  pipeline.py         the three stages tied together, settings validation
  density.py          stage 1: greyscale, contrast, simplify -> density map
  relax.py            stage 2: dot seeding, Lloyd relaxation, jitter
  drawing.py          stage 3: dot sizing, PNG/SVG output
  lines.py            outline detection for line mode
  focus.py            focus region: parsing, interactive selection, metering
sweeps/, tests/       outputs and test images (git-ignored)
```
