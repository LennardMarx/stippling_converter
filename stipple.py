"""Stippling converter - turn images into black-dot stippling drawings.

    python stipple.py photo.jpg [options]          stipple one image
    python stipple.py sweep photo.jpg [options]    every combination of settings
    python stipple.py --help / sweep --help        all options

The code lives in the stippler/ package; see README.md.
"""

import sys

from stippler import cli, sweep


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "sweep":
        sweep.main(argv[1:])
    else:
        cli.main(argv)


if __name__ == "__main__":
    main()
