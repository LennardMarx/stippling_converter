"""Parameter sweep: stipple one image with every combination of settings.

Any stipple.py option can be given as a comma-separated list; every
combination is rendered into an output directory, together with a
contact sheet (all results side by side, labelled) and a settings.csv.

Example:
    python stipple.py sweep skyline.jpg -n 20000,50000 -c 1,2 --simplify 0,0.7
    -> skyline_sweep/  (8 images + _contact_sheet.png + settings.csv)

On/off flags take yes/no lists, e.g. --variable-size no,yes
Options you don't list use the normal stipple.py defaults, except that the
output scale defaults to 1 and the seed to 1 (so results are comparable).
"""

import argparse
import csv
import itertools
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import cli, pipeline

SKIP = {"help", "input", "output", "quiet"}
NONE_LABEL = {"line_gap": "off", "focus": "off"}
LIST_SEP = {"focus": ";"}  # focus boxes contain commas, so list them with ;  # what an unset value means, for file names
SWEEP_ROOT = Path(__file__).resolve().parent.parent / "sweeps"
SWEEP_DEFAULTS = {"scale": 1.0, "seed": 1}
SHORT = {"points": "n", "iterations": "i", "dot_radius": "r", "contrast": "c",
         "white_cutoff": "w", "scale": "s", "simplify_passes": "sp",
         "variable_size": "var", "allow_touching": "touch", "work_size": "ws"}


def parse_bool(v):
    v = v.strip().lower()
    if v in ("1", "yes", "y", "true", "on"):
        return True
    if v in ("0", "no", "n", "false", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected yes/no, got {v!r}")


def build_sweep_parser(base):
    p = argparse.ArgumentParser(
        prog="stipple.py sweep",
        description="Render an image with every combination of stipple settings. "
                    "Give any option as a comma-separated list, e.g. -n 20000,50000.",
        epilog="Options not listed use stipple.py defaults (scale 1, seed 1).")
    p.add_argument("input", help="input image")
    p.add_argument("-o", "--outdir",
                   help="output directory (default: sweeps/<input name>_sweep)")
    p.add_argument("--dry-run", action="store_true",
                   help="only list the combinations")
    p.add_argument("--thumb", type=int, default=480,
                   help="thumbnail width on the contact sheet (default 480)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show per-iteration progress")
    converters = {}
    for act in base._actions:
        if act.dest in SKIP or not act.option_strings:
            continue
        is_flag = isinstance(act, argparse._StoreTrueAction)
        converters[act.dest] = parse_bool if is_flag else (act.type or str)
        default = SWEEP_DEFAULTS.get(act.dest, act.default)
        p.add_argument(*act.option_strings, dest=act.dest, metavar="LIST",
                       help=f"{act.help} [default {default}]".replace("%", "%%"))
    return p, converters


def label(name, value):
    key = SHORT.get(name, name.replace("_", "-"))
    if isinstance(value, bool):
        return key if value else f"no-{key}"
    if value is None:
        return f"{key}-{NONE_LABEL.get(name, 'auto')}"
    if name == "focus":  # boxes have commas, masks are paths -> file-name safe
        from .focus import BOX_RE
        if BOX_RE.match(str(value)):
            return key + "-".join(re.split(r"\s*[,:]\s*", str(value).strip()))
        return f"{key}-{Path(value).stem}"
    if isinstance(value, float):
        value = f"{value:g}"
    return f"{key}{value}"


def contact_sheet(entries, out, thumb_w):
    """entries: list of (png_path, caption)."""
    thumbs = []
    for path, caption in entries:
        im = Image.open(path).convert("L")
        th = round(thumb_w * im.height / im.width)
        thumbs.append((im.resize((thumb_w, th), Image.BOX), caption))
    cols = min(len(thumbs), 4)
    rows = -(-len(thumbs) // cols)
    try:
        font = ImageFont.load_default(size=max(12, thumb_w // 28))
    except TypeError:
        font = ImageFont.load_default()
    cap_h = max(18, thumb_w // 16)
    cell_h = max(t.height for t, _ in thumbs) + cap_h
    pad = 12
    sheet = Image.new("L", (cols * (thumb_w + pad) + pad, rows * (cell_h + pad) + pad), 255)
    draw = ImageDraw.Draw(sheet)
    for k, (t, caption) in enumerate(thumbs):
        x = pad + (k % cols) * (thumb_w + pad)
        y = pad + (k // cols) * (cell_h + pad)
        sheet.paste(t, (x, y))
        draw.rectangle((x - 1, y - 1, x + t.width, y + t.height), outline=180)
        draw.text((x, y + t.height + 3), caption, fill=0, font=font)
    sheet.save(out)


def main(argv=None):
    base = cli.build_parser()
    p, converters = build_sweep_parser(base)
    a = p.parse_args(argv)
    Image.MAX_IMAGE_PIXELS = None

    defaults = {act.dest: act.default for act in base._actions if act.dest != "help"}
    defaults.update(SWEEP_DEFAULTS)

    swept, fixed = {}, {}
    for name, conv in converters.items():
        raw = getattr(a, name)
        if raw is None:
            continue
        try:
            nones = ("auto", "none") if conv is parse_bool else ("auto", "none", "off")
            vals = [None if v.strip().lower() in nones else conv(v)
                    for v in raw.split(LIST_SEP.get(name, ",")) if v.strip()]
        except (ValueError, argparse.ArgumentTypeError) as e:
            p.error(f"--{name.replace('_', '-')}: {e}")
        vals = list(dict.fromkeys(vals))  # dedupe, keep order
        (swept if len(vals) > 1 else fixed)[name] = vals if len(vals) > 1 else vals[0]

    names = list(swept)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*swept.values())]

    def settings(combo):
        d = dict(defaults, **fixed, **combo)
        d.update(input=a.input, quiet=not a.verbose, svg=d.get("svg", False))
        return argparse.Namespace(**d)

    # Check every combination before doing any work, so a bad value doesn't
    # crash the sweep halfway through.
    # Interactive focus selection happens once, before the sweep.
    if "focus" in fixed and str(fixed["focus"]).strip().lower() == "select":
        from .focus import select_region
        fixed["focus"] = select_region(a.input)
        print(f"Focus region: --focus {fixed['focus']}", file=sys.stderr)
    if "focus" in swept:
        from .focus import select_region
        swept["focus"] = [select_region(a.input) if str(v).lower() == "select" else v
                          for v in swept["focus"]]
        combos = [dict(zip(names, vals)) for vals in itertools.product(*swept.values())]

    problems = sorted({e for combo in combos for e in pipeline.validate(settings(combo))})
    if problems:
        p.error("invalid settings in the sweep: " + "; ".join(problems))

    # Order so runs sharing a density map / dot set are adjacent -> reuse them.
    keyed = []
    for combo in combos:
        s = settings(combo)
        dkey = tuple(getattr(s, k) for k in pipeline.DENSITY_KEYS)
        pkey = tuple(getattr(s, k) for k in pipeline.POINTS_KEYS)
        keyed.append((str(dkey), str(pkey), combo, s, dkey, pkey))
    keyed.sort(key=lambda t: (t[0], t[1]))

    stem = Path(a.input).stem
    outdir = Path(a.outdir) if a.outdir else SWEEP_ROOT / f"{stem}_sweep"
    n_dens = len({t[0] for t in keyed})
    n_pts = len({t[1] for t in keyed})
    print(f"{len(combos)} combinations ({n_dens} density maps, {n_pts} dot "
          f"relaxations) -> {outdir}", file=sys.stderr)
    if fixed:
        print("fixed: " + ", ".join(f"{k}={v}" for k, v in fixed.items()), file=sys.stderr)
    if a.dry_run:
        for _, _, combo, *_ in keyed:
            print("  " + (" ".join(label(k, v) for k, v in combo.items()) or "(single run)"))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    cur_dkey = cur_pkey = None
    prep = pts = None
    rows, sheet_entries = [], []
    t0 = time.time()
    for k, (_, _, combo, s, dkey, pkey) in enumerate(keyed, 1):
        if dkey != cur_dkey:
            prep = pipeline.make_density(s)
            cur_dkey, cur_pkey = dkey, None
        if pkey != cur_pkey:
            pts = pipeline.make_points(s, prep)
            cur_pkey = pkey
        tag = "_".join(label(n, combo[n]) for n in names) or "single"
        out = outdir / f"{stem}_{tag}.png"
        pipeline.render(s, prep, pts, out)
        caption = "  ".join(label(n, combo[n]) for n in names) or stem
        sheet_entries.append((out, caption))
        rows.append({"file": out.name, **{n: combo[n] for n in names}})
        print(f"[{k}/{len(keyed)}] {out.name}  ({time.time() - t0:.0f}s)", file=sys.stderr)

    # Contact sheet in the original combination order (easier to read).
    order = {id(c): i for i, c in enumerate(combos)}
    idx = sorted(range(len(keyed)), key=lambda i: order[id(keyed[i][2])])
    contact_sheet([sheet_entries[i] for i in idx], outdir / "_contact_sheet.png", a.thumb)
    with open(outdir / "settings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file"] + names + list(fixed))
        w.writeheader()
        for i in idx:
            w.writerow({**rows[i], **fixed})
    print(f"Done: {len(keyed)} images + _contact_sheet.png + settings.csv in {outdir}",
          file=sys.stderr)
