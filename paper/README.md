# Paper

LaTeX source of the reproduction paper (ACM `acmart` sigconf template). The
figures are **not** stored here: they are included from [`../results/`](../results)
via `\graphicspath`, so build from a full clone of the repository, inside this
`paper/` directory.

## Building with a local TeX Live (Ubuntu/Debian)

Install the required TeX Live components (`acmart`, the ACM BibTeX style, and
the Libertine/newtx/Inconsolata fonts it uses):

```sh
sudo apt update
sudo apt install -y texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-fonts-extra \
  texlive-publishers texlive-bibtex-extra
```

Then run the usual `pdflatex → bibtex → pdflatex → pdflatex` chain:

```sh
cd paper
pdflatex -interaction=nonstopmode reproducing-confidential-llm-inference.tex
bibtex reproducing-confidential-llm-inference
pdflatex -interaction=nonstopmode reproducing-confidential-llm-inference.tex
pdflatex -interaction=nonstopmode reproducing-confidential-llm-inference.tex
```

The output is `reproducing-confidential-llm-inference.pdf` in this directory.
The extra `pdflatex` passes resolve the bibliography and cross-references; the
build is only complete when the log shows no `undefined references` warnings.

## Building with Docker (no local TeX install)

This is the route the committed PDF was built with, using the full
`texlive/texlive` image (~3 GB one-time download). Mount the **repository
root** (not `paper/`) so the figures in `../results/` are visible:

```sh
cd confidential-llms-in-tees
docker run --rm -v "$PWD":/work -w /work/paper texlive/texlive:latest bash -c "\
  pdflatex -interaction=nonstopmode reproducing-confidential-llm-inference.tex && \
  bibtex reproducing-confidential-llm-inference && \
  pdflatex -interaction=nonstopmode reproducing-confidential-llm-inference.tex && \
  pdflatex -interaction=nonstopmode reproducing-confidential-llm-inference.tex && \
  chown -R $(id -u):$(id -g) ."
```

The final `chown` is needed because the container runs as root and would
otherwise leave root-owned build outputs behind.
