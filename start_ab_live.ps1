# start_ab_live.ps1: A/B 테스트를 위한 실시간 거래를 시작합니다.

# --- FIX: PowerShell 한글 깨짐 방지 ---
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ------------------------------------

# --- 창 제목 설정 ---
$host.ui.RawUI.WindowTitle = "A/B Test Launcher"

# --- 설명 출력 ---
Write-Host "[INFO] A/B 테스트를 시작합니다. (두 개의 새 파워쉘 창이 열립니다)"
Write-Host ""
Write-Host "사전 준비:"
Write-Host "1. config/active_models_A.json 파일에 A 모델 경로 지정"
Write-Host "2. config/active_models_B.json 파일에 B 모델 경로 지정"
Write-Host ""

# --- A, B 모델 각각 별도의 파워쉘 창에서 실행 ---
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle = 'Live A'; python src/04_live_infer_ws.py --tag A --log logs/live_A.csv --active_cfg config/active_models_A.json"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle = 'Live B'; python src/04_live_infer_ws.py --tag B --log logs/live_B.csv --active_cfg config/active_models_B.json"
