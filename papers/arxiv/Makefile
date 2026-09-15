# Build ARS paper PDF with pdflatex + bibtex
MAIN = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

.PHONY: all pdf clean distclean arxiv-zip

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib figures/offline_fixture_tradeoff.png figures/offline_fixture_em.png figures/offline_fixture_f1.png
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf arxiv-source.zip

# Source zip suitable for arXiv upload (no PDF required by arXiv; they build)
arxiv-zip: $(MAIN).tex references.bib 00README.json README.md
	rm -f arxiv-source.zip
	zip -r arxiv-source.zip $(MAIN).tex references.bib figures/ 00README.json \
	  -x "*.DS_Store" "figures/.gitkeep"
	@echo "Created arxiv-source.zip"
