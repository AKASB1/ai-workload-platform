"""Draw the repository icon (docs/assets/icon.svg). Standard library only.

Usage:
    python scripts/make_assets.py

The icon is a rounded square in the area color with a simple geometric mark,
jobs on a node pool (a 2 x 2 grid of nodes, two of them busy); no text and no
third-party logo. The output is byte-stable: running the script twice gives the
same file.
"""

import os

# area colors (fill, stroke), shared with docs/diagrams/architecture.d2
FILL, INK = "#DCEBFA", "#2563A8"


def mark(ink: str) -> str:
    """SVG elements of the mark, on a 64 x 64 canvas, drawn in `ink`."""
    s = f'stroke="{ink}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" fill="none"'
    f = f'fill="{ink}"'
    return (
        f'<rect x="14" y="14" width="15" height="15" rx="3" {f}/>'
        f'<rect x="35" y="14" width="15" height="15" rx="3" {s}/>'
        f'<rect x="14" y="35" width="15" height="15" rx="3" {s}/>'
        f'<rect x="35" y="35" width="15" height="15" rx="3" {f}/>'
    )


def icon() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 64 64">'
        f'<rect x="2" y="2" width="60" height="60" rx="14" fill="{FILL}" stroke="{INK}" stroke-width="3"/>'
        f"{mark(INK)}</svg>\n"
    )


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "docs", "assets", "icon.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(icon())
    print(f"wrote {os.path.relpath(out, root)}")


if __name__ == "__main__":
    main()
