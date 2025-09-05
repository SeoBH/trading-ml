# watch_prod_log.ps1: 운영 로그를 실시간으로 모니터링합니다.

# --- FIX: PowerShell 한글 깨짐 방지 ---
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ------------------------------------

# --- 창 제목 설정 ---
$host.ui.RawUI.WindowTitle = "Watch PROD Log"

# --- 설명 출력 ---
Write-Host ""
Write-Host "[INFO] 운영 로그를 실시간으로 모니터링합니다..."
Write-Host "[INFO] 경로: logs/prod/prod.csv"
Write-Host "(Ctrl+C를 눌러 종료)"
Write-Host ""

# --- 파워쉘 명령어 실행 ---
# Get-Content: 파일 내용을 가져오는 명령어
# -Wait: 파일에 새로운 내용이 추가될 때까지 계속 대기하며 출력 (tail -f 와 동일)
# -Tail 10: 마지막 10줄을 먼저 보여주고 시작
Get-Content logs/prod/prod.csv -Wait -Tail 10
