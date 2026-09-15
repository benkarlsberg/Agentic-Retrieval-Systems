# ARS Paper (arXiv source)

**Title:** Agentic Retrieval Systems: A Reproducible Framework for Comparing Conventional RAG, Iterative, and Multi-Agent Retrieval

**Author:** Ben Karlsberg (`bkarlsberg3@gatech.edu`), Georgia Institute of Technology

This directory contains an arXiv-ready LaTeX article describing the
[Agentic Retrieval Systems](https://github.com/benkarlsberg/Agentic-Retrieval-Systems)
evaluation framework, with an **offline-fixture** validation study
(`DeterministicModel`, mini corpus, $n=13$). Numbers are harness validation /
process-overhead measurements — **not** frontier LLM benchmarks.

## Files

| Path | Role |
|------|------|
| `main.tex` | Paper source |
| `references.bib` | BibTeX bibliography |
| `figures/offline_fixture_*.png` | Plots from offline fixture runs |
| `Makefile` | Local PDF build (`pdflatex` + `bibtex`) |
| `00README.json` | File list / arXiv notes |
| `README.md` | This file |

## Build PDF locally

```bash
cd ars-paper
make pdf          # produces main.pdf
# or:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires a TeX distribution with `pdflatex`, `bibtex`, and packages:
`graphicx`, `booktabs`, `hyperref`, `natbib`, `amsmath`, `geometry`, etc.
(TeX Live recommended.)

## arXiv upload instructions

1. **Prepare source archive** (preferred: let arXiv compile from source):
   ```bash
   make arxiv-zip
   # or manually:
   tar czvf ars-arxiv.tar.gz main.tex references.bib figures/ 00README.json
   ```
   Include **all** of: `main.tex`, `references.bib`, and the three PNGs under `figures/`.

2. **Go to** [https://arxiv.org/submit](https://arxiv.org/submit) → start a new submission.

3. **Categories (suggested):**
   - Primary: **cs.IR** (Information Retrieval)
   - Cross-lists: **cs.CL**, **cs.AI**

4. **Comments field (suggested):**
   ```
   Systems/evaluation framework paper with offline-fixture validation
   (DeterministicModel, n=13). Not a frontier LLM benchmark. Code:
   https://github.com/benkarlsberg/Agentic-Retrieval-Systems
   ```

5. **Abstract:** paste from `\begin{abstract}` in `main.tex` (plain text; remove TeX macros as needed).

6. **License:** choose a license consistent with your preferences and the MIT-licensed code release.

7. After processing completes, download the arXiv-built PDF and verify figures, bibliography, and that the offline-fixture disclaimer is visible in the abstract/intro.

### Honesty checklist before submit

- [ ] Abstract and Section 1 state that results are offline-fixture / synthetic harness validation
- [ ] No invented HotpotQA/NQ leaderboard numbers
- [ ] No SOTA quality claims from fixture EM/F1 (quality matched; signal is cost/process)
- [ ] Author line is Ben Karlsberg only
- [ ] Code URL is correct

## Copying to a Mac

From the machine that has `/workspace/ars-paper/`:

```bash
# example: scp/rsync the whole directory, or the zip
rsync -av ars-paper/ /path/on/mac/ars-paper/
# or
cp arxiv-source.zip ~/Downloads/
```

## Citation (preprint placeholder)

```bibtex
@article{karlsberg2026ars,
  title={Agentic Retrieval Systems: A Reproducible Framework for Comparing Conventional {RAG}, Iterative, and Multi-Agent Retrieval},
  author={Karlsberg, Ben},
  year={2026},
  note={arXiv preprint; offline-fixture validation}
}
```

Replace with the arXiv id after acceptance of the submission.
