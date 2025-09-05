# promote_winner.ps1: A/B 테스트 승자를 새로운 챔피언으로 승격시킵니다.

# --- FIX: PowerShell 한글 깨짐 방지 ---
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ------------------------------------

# --- 설명 출력 ---
$host.ui.RawUI.WindowTitle = "Promote Winner"

# --- 설명 출력 ---
Write-Host ""
Write-Host "[INFO] A/B Test 승자 모델을 PROD 모델로 승격합니다..."
Write-Host "(logs/ab_score.txt 파일을 읽어 config/active_models.json 을 업데이트)"
Write-Host ""

# --- 파이썬 스크립트 실행 ---
python src/08_promote_winner.py

# --- 실행 결과 확인 ---
if ($LASTEXITCODE -ne 0) {
    Write-Host "" -ForegroundColor Red
    Write-Host "[ERROR] 모델 승격 중 오류가 발생했습니다." -ForegroundColor Red
} else {
    Write-Host "" -ForegroundColor Green
    Write-Host "[SUCCESS] 모델 승격이 완료되었습니다." -ForegroundColor Green
}

# --- 잠시 대기 ---
Write-Host ""
Read-Host -Prompt "Press Enter to continue"
