# score_ab.ps1: A/B 테스트 결과를 채점합니다.

# --- FIX: PowerShell 한글 깨짐 방지 ---
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ------------------------------------

# --- 창 제목 설정 ---
$host.ui.RawUI.WindowTitle = "Score A/B Test"

# --- 설명 출력 ---
Write-Host ""
Write-Host "[INFO] A/B 테스트 결과를 채점합니다..."
Write-Host "(A모델과 B모델의 실시간 거래 로그를 비교하여 승자를 결정)"
Write-Host ""

# --- 파이썬 스크립트 실행 ---
python src/07_score_ab.py logs/live_A.csv logs/live_B.csv

# --- 실행 결과 확인 ---
if ($LASTEXITCODE -ne 0) {
    Write-Host "" -ForegroundColor Red
    Write-Host "[ERROR] A/B 테스트 채점 중 오류가 발생했습니다." -ForegroundColor Red
} else {
    Write-Host "" -ForegroundColor Green
    Write-Host "[SUCCESS] A/B 테스트 채점이 완료되었습니다." -ForegroundColor Green
}

# --- 잠시 대기 ---
Write-Host ""
Read-Host -Prompt "Press Enter to continue"
