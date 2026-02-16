$ErrorActionPreference = "Stop"

# [핵심] 파이썬 인코딩 환경변수를 여기서 안전하게 설정 (외부 set 명령어 필요 없음)
$env:PYTHONUTF8 = "1"

Start-Transcript -Path "training_log.txt" -Append

Write-Host "========================================================" -ForegroundColor Green
Write-Host "   🚀 Pipeline Started (Fixed Version)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green

# [Step 0]
Write-Host "Running Step 0..." -ForegroundColor Cyan
python 0_preprocess_cases.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 0 Failed"; Stop-Transcript; exit 1 }

# [Step 1]
Write-Host "Running Step 1..." -ForegroundColor Cyan
python 1_build_datasets.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 1 Failed"; Stop-Transcript; exit 1 }

# [Step 2]
Write-Host "Running Step 2 (Training)..." -ForegroundColor Cyan
python 2_train_system.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 2 Failed"; Stop-Transcript; exit 1 }

# [Step 3]
Write-Host "Running Step 3 (Evaluation)..." -ForegroundColor Cyan
python 3_eval_system.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 3 Failed"; Stop-Transcript; exit 1 }

Write-Host "========================================================" -ForegroundColor Green
Write-Host "   🎉 All Jobs Done. Check summary.txt" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green

Stop-Transcript