#!/bin/sh
# Fetch the raw data, run every experiment, regenerate tables and number macros.  About 25 minutes on one core.
set -e
cd "$(dirname "$0")"
python3 code/fetch_data.py
python3 code/record_environment.py
python3 code/extract_semcor.py
python3 code/dna_experiment.py
python3 code/dna_identification.py
python3 code/embedding_demo.py
python3 code/semcor_experiment.py
python3 code/semcor_subsample.py
python3 code/inventory_experiment.py
python3 code/verification.py
python3 code/estimator_validation.py
python3 code/make_assets.py
echo ALL_DONE
