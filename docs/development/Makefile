LATEXMK ?= latexmk
PYTHON ?= python3
FLAGS := -xelatex -interaction=nonstopmode -file-line-error -halt-on-error

.PHONY: all qa clean

all:
	$(LATEXMK) $(FLAGS) -jobname=development-log -outdir=build main.tex
	mkdir -p output/pdf
	cp build/development-log.pdf output/pdf/development-log.pdf

qa: all
	$(PYTHON) tools/verify.py

clean:
	$(LATEXMK) -C -jobname=development-log -outdir=build main.tex
