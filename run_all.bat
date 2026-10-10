@echo off
REM run_all.bat -- Fetch raw data, run all experiments, and regenerate tables/assets on Windows Command Prompt.
cd /d "%~dp0"

echo [1/6] Fetching data and environment...
python code\fetch_data.py || exit /b %errorlevel%
python code\record_environment.py || exit /b %errorlevel%
python code\extract_semcor.py || exit /b %errorlevel%

echo [2/6] Running DNA experiments...
python code\dna_experiment.py || exit /b %errorlevel%
python code\dna_identification.py || exit /b %errorlevel%
python code\embedding_demo.py || exit /b %errorlevel%

echo [3/6] Running SemCor experiments...
python code\semcor_experiment.py || exit /b %errorlevel%
python code\semcor_subsample.py || exit /b %errorlevel%
python code\inventory_experiment.py || exit /b %errorlevel%

echo [4/6] Running verification and estimator validation...
python code\verification.py || exit /b %errorlevel%
python code\estimator_validation.py || exit /b %errorlevel%

echo [5/6] Generating assets...
python code\make_assets.py || exit /b %errorlevel%

echo [6/6] ALL_DONE
