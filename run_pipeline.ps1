# 에러나면 즉시 멈춤
$ErrorActionPreference = "Stop"

# 파이썬 인코딩 설정 (여기서 안전하게 1로 고정)
$env:PYTHONUTF8 = "1"

# 로그 기록 시작
Start-Transcript -Path "training_log.txt" -Append

Write-Host "------------------------------------------------"
Write-Host "   Pipeline Started"
Write-Host "------------------------------------------------"

# [Step 0]
Write-Host "Running Step 0 (Preprocess)..."
python 0_preprocess_cases.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 0 Failed"; Stop-Transcript; exit 1 }

# [Step 1]
Write-Host "Running Step 1 (Build Dataset)..."
python 1_build_datasets.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 1 Failed"; Stop-Transcript; exit 1 }

# [Step 2]
Write-Host "Running Step 2 (Training - 16 Hours)..."
python 2_train_system.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 2 Failed"; Stop-Transcript; exit 1 }

# [Step 3]
Write-Host "Running Step 3 (Evaluation)..."
python 3_eval_system.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 3 Failed"; Stop-Transcript; exit 1 }

Write-Host "------------------------------------------------"
Write-Host "   ALL DONE! Check summary.txt"
Write-Host "------------------------------------------------"

Stop-Transcript