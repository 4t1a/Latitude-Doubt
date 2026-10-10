# run_all.ps1 -- Fetch raw data, run all experiments, and regenerate tables/assets on Windows PowerShell.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "==> Fetching raw data..." -ForegroundColor Cyan
python code/fetch_data.py

Write-Host "==> Recording environment..." -ForegroundColor Cyan
python code/record_environment.py

Write-Host "==> Extracting SemCor..." -ForegroundColor Cyan
python code/extract_semcor.py

Write-Host "==> Running DNA experiments..." -ForegroundColor Cyan
python code/dna_experiment.py
python code/dna_identification.py
python code/embedding_demo.py

Write-Host "==> Running SemCor experiments..." -ForegroundColor Cyan
python code/semcor_experiment.py
python code/semcor_subsample.py
python code/inventory_experiment.py

Write-Host "==> Running verification and estimator validation..." -ForegroundColor Cyan
python code/verification.py
python code/estimator_validation.py

Write-Host "==> Generating LaTeX tables and assets..." -ForegroundColor Cyan
python code/make_assets.py

Write-Host "==> ALL_DONE: All experiments reproduced successfully!" -ForegroundColor Green
