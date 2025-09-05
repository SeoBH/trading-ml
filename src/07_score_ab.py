# 07_score_ab.py: A/B 테스트 로그 성과 비교 스크립트
# A모델과 B모델이 각각 남긴 실시간 거래 로그 파일을 읽어와서,
# 어떤 모델의 성과가 더 좋았는지 비교하고 승자를 결정합니다.

import pandas as pd
import numpy as np
import sys
import os

def load_log(path):
    """로그 파일을 불러오는 함수"""
    try:
        df=pd.read_csv(path)
        # equity 컬럼이 숫자가 아닌 경우를 대비해, 에러 발생 시 숫자가 아닌 값으로 처리
        df["equity"]=pd.to_numeric(df["equity"],errors="coerce")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Log file not found at: {path}")
        # 빈 데이터프레임을 생성하여 프로그램이 중단되지 않도록 함
        return pd.DataFrame(columns=['equity'])

def metrics(df):
    """자산 곡선(equity) 데이터로 주요 성과 지표를 계산하는 함수"""
    if df.empty or 'equity' not in df.columns or df['equity'].dropna().empty:
        return dict(final=1.0, mdd=1.0, sharpe=0.0, trades=0)

    eq = df["equity"].dropna().values
    
    # 1. 최종 자산 (Final Equity)
    final_equity = eq[-1] if len(eq) > 0 else 1.0
    
    # 2. 최대 낙폭 (MDD, Maximum Drawdown)
    # (가장 높았을 때 자산 - 가장 낮았을 때 자산) / 가장 높았을 때 자산
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    mdd = np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    # 3. 샤프 지수 (Sharpe Ratio)
    # (수익률의 평균) / (수익률의 표준편차). 위험 대비 수익성을 나타내는 지표.
    # 여기서는 일일 샤프 지수를 연율화하여 계산 (근사치)
    ret = pd.Series(eq).pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * np.sqrt(252*24*60)) if ret.std() > 0 else 0

    # 4. 총 거래 횟수 (Total Trades)
    trades = df[df['action'] == 'BUY'].shape[0] if 'action' in df.columns else 0
    
    return dict(final=final_equity, mdd=mdd, sharpe=sharpe, trades=trades)

# --- 메인 로직 ---
if __name__=="__main__":
    # 터미널에서 실행할 때, `python 07_score_ab.py [A로그경로] [B로그경로]` 와 같이 인자를 받습니다.
    if len(sys.argv)<3:
        print("Usage: python 07_score_ab.py [log_A_path] [log_B_path]")
        sys.exit(1)
    
    log_a_path, log_b_path = sys.argv[1], sys.argv[2]
    
    # A, B 로그를 불러와 각각의 성과 지표를 계산합니다.
    metrics_A = metrics(load_log(log_a_path))
    metrics_B = metrics(load_log(log_b_path))
    
    print("--- A/B Test Score ---")
    print(f"[Model A]: {metrics_A}")
    print(f"[Model B]: {metrics_B}")
    print("----------------------")
    
    # 최종 자산(final)을 기준으로 승자를 결정합니다.
    winner = "A" if metrics_A["final"] > metrics_B["final"] else "B"
    
    # 승자 정보를 logs/ab_score.txt 파일에 기록합니다.
    os.makedirs("logs", exist_ok=True)
    with open("logs/ab_score.txt","w") as f: 
        f.write(winner)
        
    print(f"[WINNER] Model {winner}")