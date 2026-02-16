# 1. 실행할 스크립트 리스트 정의
$scripts = @(
    "0_preprocess_cases.py",
    "1_build_datasets.py",
    "2_train_system.py",
    "3_eval_system.py"
)

$total = $scripts.Count
$currentIndex = 0

Write-Host "`n[ ML Pipeline Automation Start ]" -ForegroundColor Cyan

foreach ($script in $scripts) {
    $currentIndex++
    $percent = [int](($currentIndex / $total) * 100)
    
    # 상단 상태바 업데이트 (이 기능이 로그가 쌓이는 것을 방지함)
    Write-Progress -Activity "Pipeline Processing" -Status "Running: $script" -PercentComplete $percent
    
    # 현재 실행 중인 스크립트 강조 표시
    Clear-Host
    Write-Host "==========================================" -ForegroundColor Gray
    Write-Host "  CURRENT TASK: $script ($currentIndex / $total)" -ForegroundColor Yellow
    Write-Host "  PROGRESS: $percent%" -ForegroundColor Yellow
    Write-Host "==========================================" -ForegroundColor Gray
    Write-Host "`nRunning script output below..." -ForegroundColor DarkGray

    # 스크립트 실행 및 종료 코드 확인
    python $script
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Error occurred in $script. Pipeline stopped." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# 모든 작업 완료 후 처리
Clear-Host
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  ALL TASKS COMPLETED SUCCESSFULLY!       " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "`nSystem will shutdown in 60 seconds." -ForegroundColor Cyan

# 60초 후 종료 예약
shutdown /s /t 60