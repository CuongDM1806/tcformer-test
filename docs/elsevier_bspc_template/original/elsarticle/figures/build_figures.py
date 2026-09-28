#!/usr/bin/env python3
"""Build the paper's vector architecture figures from their TikZ sources."""

from pathlib import Path
import subprocess


FIGURE_DIR = Path(__file__).resolve().parent
FIGURES = (
    "architecture_overview.tex",
    "encoder_detail.tex",
    "adaptation_objective.tex",
)


def main() -> None:
    for source_name in FIGURES:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                source_name,
            ],
            cwd=FIGURE_DIR,
            check=True,
        )
        print(f"built {FIGURE_DIR / source_name.replace('.tex', '.pdf')}")


if __name__ == "__main__":
    main()
