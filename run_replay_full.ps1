# run_replay_full.ps1: 전체 과거 데이터에 대해 정밀 리플레이를 실행합니다.

# --- FIX: PowerShell 한글 깨짐 방지 ---
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ------------------------------------

# --- 창 제목 설정 ---
$host.ui.RawUI.WindowTitle = "Full Replay for Retraining"

# --- 설명 출력 ---
Write-Host "" -ForegroundColor Yellow
Write-Host "[WARNING] 전체 데이터셋에 대한 정밀 리플레이를 시작합니다."
Write-Host "[WARNING] 이 작업은 매우 오래 걸립니다 (10시간 이상 소요될 수 있음). 절전 모드를 비활성화했는지 확인하세요."
Write-Host "" -ForegroundColor Yellow

# --- 파이썬 스크립트 실행 ---
# 트렌드 피처가 포함된 새로운 데이터셋을 사용합니다.
$env:PYTHONUTF8=1 # <-- FIX: 파이썬 출력 한글 깨짐 방지
python src/04_live_infer_ws.py --replay data/dataset_1m_H60_with_trends.csv --tag REPLAY_FULL --log logs/replay/replay_full.csv

# --- 실행 결과 확인 ---
if ($LASTEXITCODE -ne 0) {
    Write-Host "" -ForegroundColor Red
    Write-Host "[ERROR] 전체 리플레이 실행 중 오류가 발생했습니다." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to continue"
} else {
    Write-Host "" -ForegroundColor Green
    Write-Host "[SUCCESS] 전체 리플레이가 성공적으로 완료되었습니다." -ForegroundColor Green
}