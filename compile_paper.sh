#!/bin/bash

# Script to compile the accumux academic paper
# Requires: pdflatex or xelatex installed

echo "Compiling accumux academic paper..."

# Check if pdflatex is available
if command -v pdflatex &> /dev/null; then
    echo "Using pdflatex..."
    pdflatex -interaction=nonstopmode accumux_paper.tex
    pdflatex -interaction=nonstopmode accumux_paper.tex  # Second pass for references
    echo "Paper compiled successfully! Output: accumux_paper.pdf"
elif command -v xelatex &> /dev/null; then
    echo "Using xelatex..."
    xelatex -interaction=nonstopmode accumux_paper.tex
    xelatex -interaction=nonstopmode accumux_paper.tex  # Second pass for references
    echo "Paper compiled successfully! Output: accumux_paper.pdf"
else
    echo "Error: Neither pdflatex nor xelatex found."
    echo "Please install a LaTeX distribution (e.g., texlive-full on Ubuntu/Debian)"
    echo ""
    echo "Installation suggestions:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  macOS: brew install --cask mactex"
    echo "  Windows: Install MiKTeX from https://miktex.org/"
    exit 1
fi

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f accumux_paper.aux accumux_paper.log accumux_paper.out accumux_paper.bbl accumux_paper.blg

echo "Done!"