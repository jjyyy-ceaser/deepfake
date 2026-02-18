$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Start-Transcript -Path "training_log.txt" -Append
Write-Host "🚀 Pipeline Start..." -ForegroundColor Green
python 0_preprocess_cases.py
python 1_build_datasets.py
python 2_train_system.py
python 3_eval_system.py
Write-Host "🎉 Pipeline Success!" -ForegroundColor Green
Stop-Transcript