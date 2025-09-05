# 03_backtest_ml.py: 모델 기반 백테스트 스크립트
# 학습이 완료된 모델을 불러와, 과거 데이터 전체에 대해 거래 시뮬레이션을 실행합니다.
# 이를 통해 "이 모델을 과거에 사용했다면 어떤 성과를 냈을까?"를 빠르게 확인할 수 있습니다.

import pandas as pd
import numpy as np
from joblib import load

# === 설정값 ===
# 백테스트에 필요한 파일 경로들을 지정합니다.
DATA = "data/BTCUSDT-1m-202401_202507.csv"      # 원본 1분봉 데이터
DATASET = "data/dataset_1m_H60_with_trends.csv" # 피처가 포함된 데이터셋
MODEL_REG = "models/lgbm_retH_trends.pkl"         # 사용할 회귀 모델
MODEL_CLS = "models/lgbm_cls_trends.pkl"         # 사용할 분류 모델

# 시뮬레이션에 사용할 거래 파라미터
H = 60              # 포지션 보유 시간 (60분)
FEE = 0.0003        # 수수료 (0.03%)
SLIP = 0.0005       # 슬리피지 (체결 오차)
THR_REG = 0.0003    # 회귀 모델 진입 임계값 (예측 수익률 0.03% 이상)
THR_CLS = 0.55      # 분류 모델 진입 임계값 (익절 확률 55% 이상)
RISK = 1.0          # 한 번에 투자할 자산의 비율 (1.0 = 100%)

# --- 메인 로직 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    # 원본 가격 데이터와, 피처/레이블이 포함된 데이터셋을 모두 불러옵니다.
    raw = pd.read_csv(DATA, parse_dates=['datetime']).sort_values('datetime').set_index('datetime')
    ds  = pd.read_csv(DATASET, parse_dates=['datetime']).sort_values('datetime').set_index('datetime')

    # 2. 피처(X) 준비
    # 데이터셋에서 레이블(정답) 컬럼들을 제외하여, 모델에 입력할 피처 데이터만 남깁니다.
    feature_cols = [c for c in ds.columns if c not in ['ret_H', 'y_tp_sl']]
    X = ds[feature_cols]

    # 3. 모델 로드 및 예측
    # 저장해둔 회귀/분류 모델을 불러옵니다.
    reg = load(MODEL_REG)
    cls = load(MODEL_CLS)

    # 피처 데이터를 모델에 넣어 전체 기간에 대한 예측을 한 번에 수행합니다.
    print("Predicting with models...")
    pred_ret = pd.Series(reg.predict(X), index=X.index, name='pred_ret')
    proba    = pd.Series(cls.predict_proba(X)[:,1], index=X.index, name='proba')

    # 4. 백테스트 시뮬레이션
    # 체결 가격은 보수적으로, 신호가 발생한 다음 캔들의 시가(open)로 가정합니다.
    price = raw['close']
    next_price = raw['open'].shift(-1)

    def run_bt(score, thr):
        """예측 점수(score)와 임계값(thr)을 받아 백테스트를 실행하는 함수"""
        # 진입 신호가 발생한 시점들을 찾습니다.
        entries = score[score >= thr].index
        
        # 포트폴리오 변수 초기화
        cash = 1.0      # 초기 자본금
        pos = 0.0       # 현재 보유 수량
        nav_history = [] # 매 시점의 자산(NAV)을 기록할 리스트
        exit_t = pd.NaT # 청산 시간을 기록할 변수

        # 전체 데이터 기간을 처음부터 끝까지 순회합니다.
        for t in price.index:
            current_price = price.loc[t]
            
            # --- 진입 로직 ---
            # 진입 시점에 해당하고, 현재 포지션이 없고, 다음 캔들 가격이 유효할 때
            if t in entries and pos == 0.0 and pd.notna(next_price.loc[t]):
                entry_px = next_price.loc[t] * (1 + FEE + SLIP) # 비용을 고려한 진입 가격
                qty = cash / entry_px * RISK # 현재 현금으로 살 수 있는 수량
                pos = qty
                cash -= qty * entry_px
                exit_t = t + pd.Timedelta(minutes=H) # H분 뒤를 청산 시간으로 설정
            
            # --- 청산 로직 ---
            # 포지션을 보유 중이고, 현재 시간이 설정된 청산 시간이거나 그 이후일 때
            if pos > 0 and t >= exit_t and pd.notna(next_price.loc[t]):
                exit_px = next_price.loc[t] * (1 - FEE - SLIP) # 비용을 고려한 청산 가격
                cash += pos * exit_px
                pos = 0.0
            
            # --- 자산 평가 ---
            # 매 시점의 총 자산(NAV: Net Asset Value)을 계산합니다.
            equity = cash + (pos * current_price if pos > 0 else 0)
            nav_history.append((t, equity))
        
        # 자산 기록을 데이터프레임으로 변환하여 반환
        nav_df = pd.DataFrame(nav_history, columns=['datetime','equity']).set_index('datetime')
        return nav_df

    # 5. 결과 계산 및 저장
    print("Running backtests...")
    # 회귀 모델과 분류 모델 각각에 대해 백테스트를 실행합니다.
    nav_reg = run_bt(pred_ret, THR_REG)
    nav_cls = run_bt(proba, THR_CLS)

    # 결과를 CSV 파일로 저장합니다.
    os.makedirs("backtest", exist_ok=True)
    out1 = "backtest/nav_reg.csv"; out2 = "backtest/nav_cls.csv"
    nav_reg.to_csv(out1); nav_cls.to_csv(out2)
    
    print(f"Backtest results saved to: {out1}, {out2}")
    print(f"REG Model Final Equity: {nav_reg['equity'].iloc[-1]:.4f}")
    print(f"CLS Model Final Equity: {nav_cls['equity'].iloc[-1]:.4f}")