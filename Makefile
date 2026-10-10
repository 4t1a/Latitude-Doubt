# make pdf      : build the paper from the shipped results
# make all      : fetch data, rerun every experiment, regenerate tables and figures' data, build the paper
PY ?= python3
.PHONY: all data results assets pdf clean
all: data results assets pdf
data:
	$(PY) code/fetch_data.py
	$(PY) code/record_environment.py
	$(PY) code/extract_semcor.py
results:
	$(PY) code/dna_experiment.py
	$(PY) code/dna_identification.py
	$(PY) code/embedding_demo.py
	$(PY) code/semcor_experiment.py
	$(PY) code/semcor_subsample.py
	$(PY) code/inventory_experiment.py
	$(PY) code/verification.py
	$(PY) code/estimator_validation.py
assets:
	$(PY) code/make_assets.py
pdf:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
clean:
	rm -f main.aux main.log main.out main.bbl main.blg main.toc
