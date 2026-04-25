"""Save a matplotlib Figure in publication-ready formats.

For each figure we produce three files (basename.{png,pdf,tex}):

    <basename>.png   raster preview
    <basename>.pdf   vector figure — what goes into the paper
    <basename>.tex   self-contained LaTeX wrapper that \\includegraphics
                     the PDF; compiles with `pdflatex <basename>.tex`
                     to a tight-bordered standalone PDF, or can be
                     \\input directly into the paper

Usage:
    save_png_pdf_tex(fig, "/path/to/plots/figname")
"""
from __future__ import annotations
import os
import matplotlib


matplotlib.rcParams.update({
    "font.family":          "serif",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "legend.fontsize":      9,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "pdf.fonttype":         42,    # editable text in PDFs
    "ps.fonttype":          42,
})


_STANDALONE_TPL = r"""\documentclass[border=2pt]{standalone}
\usepackage{graphicx}
\begin{document}
\includegraphics{%s.pdf}
\end{document}
"""


def save_png_pdf_tex(fig, basename, dpi=200):
    """Save fig as <basename>.png, <basename>.pdf, and <basename>.tex.

    The TeX file is a thin standalone wrapper. To embed the figure in a
    paper, `\\input{<basename>.tex}` inside a figure environment, or
    \\includegraphics the PDF directly.
    """
    png = basename + ".png"
    pdf = basename + ".pdf"
    tex = basename + ".tex"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    with open(tex, "w") as f:
        f.write(_STANDALONE_TPL % os.path.basename(basename))
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {tex}")
