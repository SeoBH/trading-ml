# start_prod_live.ps1: 운영 환경에서 실시간 거래를 시작합니다.

# --- FIX: PowerShell 한글 깨짐 방지 ---
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ------------------------------------

# --- 창 제목 설정 ---
$host.ui.RawUI.WindowTitle = "PROD Live Trading"

# --- 설명 출력 ---
Write-Host ""
Write-Host "[INFO] 운영(PROD) 모델 실시간 거래를 시작합니다..."
Write-Host "(사용할 모델은 config/active_models.json 에 의해 결정됩니다.)"
Write-Host ""

# --- 파이썬 스크립트 실행 ---
python src/04_live_infer_ws.py --tag PROD --log logs/prod/prod.csv --active_cfg config/active_models.json

# --- 실행 결과 확인 ---
if ($LASTEXITCODE -ne 0) {
    Write-Host "" -ForegroundColor Red
    Write-Host "[ERROR] 실시간 거래 중 오류가 발생했습니다." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to continue"
}
