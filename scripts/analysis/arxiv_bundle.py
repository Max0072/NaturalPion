"""Build the arXiv submission from docs/paper/latex: a flat directory and its .tar.gz, plus the abstract as plain
text for the submission form.

    python scripts/analysis/render_tex.py && python scripts/analysis/arxiv_bundle.py

Writes docs/paper/arxiv/ (git-ignored; rebuild it rather than edit it):
  src/                 main.tex, numbers.tex, main.bbl, the three figures, the ICML 2026 style files
  natural-pion.tar.gz  src/ packed, to upload
  abstract.txt         the abstract with every number filled in, LaTeX reduced to what arXiv's form renders
  comments.txt         the form's Comments field: pages, figures, the code link
  check.pdf            src/ compiled from scratch, to compare with docs/paper/latex/main.pdf

arXiv compiles the source itself and runs no BibTeX, so the bibliography goes in as main.bbl. Comments are
stripped because arXiv publishes the source. Figures are copied next to main.tex because \\graphicspath points
outside the directory, which the archive does not contain.
"""
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "docs/paper/latex"
FIG = ROOT / "docs/paper/figures/preprint"
OUT = ROOT / "docs/paper/arxiv"
STYLE = ["icml2026.sty", "icml2026.bst", "fancyhdr.sty", "algorithm.sty", "algorithmic.sty"]


def strip_comments(text: str) -> str:
    out, verb = [], False
    for line in text.split("\n"):
        if "\\begin{verbatim}" in line:
            verb = True
        if not verb:
            m = re.search(r"(?<!\\)%", line)
            if m:
                line = line[:m.start()].rstrip()
                if not line:
                    continue                                     # a comment-only line goes entirely
        if "\\end{verbatim}" in line:
            verb = False
        out.append(line)
    return "\n".join(out)


def tectonic(src: Path, *args):
    subprocess.run(["tectonic", "-X", "compile", *args, "main.tex"], cwd=src, check=True,
                   capture_output=True, text=True)


def plain_abstract(tex: str, numbers: dict) -> str:
    a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S).group(1)
    a = re.sub(r"\\pn\{([^}]*)\}\{((?:[^{}\\]|\\%)*)\}", lambda m: numbers[(m.group(1), m.group(2))], a)
    a = a.replace("\\(", "$").replace("\\)", "$")
    a = re.sub(r"\\ensuremath\{(.*?)\}(?=[^}]|$)", r"$\1$", a)
    a = re.sub(r"\\textbf\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\url\{([^}]*)\}", r"\1", a)                          # the form links bare URLs itself
    a = a.replace("\\%", "%").replace("\\,", " ").replace("--", "-").replace("~", " ")
    paras = [" ".join(p.split()) for p in a.strip().split("\n\n")]
    return "\n\n".join(paras) + "\n"


def main():
    if OUT.exists():
        shutil.rmtree(OUT / "src", ignore_errors=True)
    src = OUT / "src"
    src.mkdir(parents=True)

    # the bibliography, from a build of the current source
    build = OUT / "build"                       # compiled in place (the figures path is relative), output kept apart
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir()
    tectonic(TEX, "--keep-intermediates", "--outdir", str(build))
    shutil.copy(build / "main.bbl", src / "main.bbl")
    shutil.rmtree(build)

    tex = (TEX / "main.tex").read_text()
    assert "\\graphicspath{{../figures/preprint/}}" in tex
    tex = tex.replace("\\graphicspath{{../figures/preprint/}}\n", "")
    # the bibliography read straight from main.bbl: no BibTeX run anywhere, so what is checked is what arXiv builds
    bib = "\\bibliography{references}\n\\bibliographystyle{icml2026}"
    assert bib in tex
    tex = tex.replace(bib, "\\input{main.bbl}")
    (src / "main.tex").write_text(strip_comments(tex))
    (src / "numbers.tex").write_text(strip_comments((TEX / "numbers.tex").read_text()) + "\n")
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
        shutil.copy(FIG / name, src / name)
    for name in STYLE:
        shutil.copy(TEX / name, src / name)

    # compile what will be uploaded, from nothing but itself
    check = OUT / "check"
    shutil.rmtree(check, ignore_errors=True)
    shutil.copytree(src, check)
    tectonic(check, "--keep-logs")
    shutil.copy(check / "main.pdf", OUT / "check.pdf")
    pages = int(re.search(r"Output written on main\.(?:pdf|xdv) \((\d+) page", (check / "main.log").read_text()).group(1))
    shutil.rmtree(check)

    with tarfile.open(OUT / "natural-pion.tar.gz", "w:gz") as tar:
        for f in sorted(src.iterdir()):
            tar.add(f, arcname=f.name)

    defs = dict(((k, f), v) for k, f, v in
                re.findall(r"\\pndef\{([^}]*)\}\{((?:[^{}\\]|\\%)*)\}\{(.*)\}$", (TEX / "numbers.tex").read_text(), flags=re.M))
    ab = plain_abstract(tex, defs)
    (OUT / "abstract.txt").write_text(ab)
    # the form's Comments field: pages, figures and the code link (the link is in the paper's introduction)
    figs = len(re.findall(r"\\begin\{figure\*?\}", tex))
    url = re.search(r"\\url\{(https://github\.com/[^}]*)\}", tex).group(1)
    (OUT / "comments.txt").write_text(f"{pages} pages, {figs} figures. Code and run records: {url}\n")
    print("files:", sorted(f.name for f in src.iterdir()))
    print(f"tar: {(OUT / 'natural-pion.tar.gz').stat().st_size / 1024:.0f} KiB; abstract: {len(ab.strip())} characters "
          "(arXiv's limit is 1920)")


if __name__ == "__main__":
    main()
