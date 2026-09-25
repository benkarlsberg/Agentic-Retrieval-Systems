# Build ARS paper PDF with pdflatex + bibtex
MAIN = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex
PYTHON ?= python3
ARS_ROOT ?= ../..
FIGURES = figures/live_pareto_tokens.png figures/live_category_em.png

.PHONY: all pdf analysis clean distclean arxiv-zip

all: pdf

pdf: $(MAIN).pdf

# pdflatex -> bibtex -> pdflatex -> pdflatex; also produces $(MAIN).bbl
$(MAIN).pdf $(MAIN).bbl: $(MAIN).tex references.bib $(FIGURES)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Recompute analysis/results.json, analysis/tables.tex and the figures from
# the run artifacts (needs numpy + matplotlib).
analysis:
	$(PYTHON) analysis/compute_tables.py --ars-root $(ARS_ROOT)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf arxiv-source.zip

# Source zip for arXiv upload. Includes the generated main.bbl so arXiv does
# not need to run bibtex. The PDF is not included.
arxiv-zip: $(MAIN).tex $(MAIN).bbl references.bib $(FIGURES) 00README.json
	rm -f arxiv-source.zip
	zip -r arxiv-source.zip $(MAIN).tex $(MAIN).bbl references.bib figures/ 00README.json \
	  -x "*.DS_Store" "figures/.gitkeep"
	@echo "Created arxiv-source.zip"
