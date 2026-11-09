#!/usr/bin/env bash
# Render docs/diagrams/*.d2 to docs/figures/*.svg with D2 (https://d2lang.com).
# Needs the d2 binary on PATH (or D2=/path/to/d2). CI does not run this script.
# The layout engine, sketch mode, and padding are set inside each .d2 file.
set -euo pipefail
cd "$(dirname "$0")/.."
d2bin="${D2:-d2}"
mkdir -p docs/figures
for src in docs/diagrams/*.d2; do
  out="docs/figures/$(basename "${src%.d2}").svg"
  "$d2bin" "$src" "$out"
  # keep the SVG free of tool-version metadata
  sed -i.bak 's/ data-d2-version="[^"]*"//' "$out" && rm -f "$out.bak"
  echo "rendered $out"
done
