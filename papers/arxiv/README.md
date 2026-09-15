# Paper sources

**Title:** Agentic Retrieval Systems: A Reproducible Framework for Comparing Conventional RAG, Iterative, and Multi-Agent Retrieval

**Author:** Ben Karlsberg (`bkarlsberg3@gatech.edu`), Georgia Institute of Technology

## Build

```bash
cd papers/arxiv
make pdf
```

Requires TeX Live (or similar) with `pdflatex` and `bibtex`.

## Contents

| File | Description |
|------|-------------|
| `main.tex` | Paper source |
| `references.bib` | Bibliography |
| `figures/` | Figures |
| `main.pdf` | Built PDF |
| `arxiv-source.zip` | Source bundle for arXiv |

Primary category suggestion: `cs.IR` (cross-list `cs.CL`, `cs.AI`).

The empirical section reports offline fixture validation with a deterministic model on a small corpus (`n=13`). Live-model experiments, when available, should be labeled separately.
