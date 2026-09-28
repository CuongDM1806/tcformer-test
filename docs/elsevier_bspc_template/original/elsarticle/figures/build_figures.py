#!/usr/bin/env python3
"""Build the paper's vector architecture figures.

The figures are drawn by make_figures.py (matplotlib, no LaTeX required).
The *.tex TikZ sources in this folder are the superseded first drafts.
"""

from pathlib import Path
import subprocess
import sys


FIGURE_DIR = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run(
        [sys.executable, "make_figures.py", *sys.argv[1:]],
        cwd=FIGURE_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
