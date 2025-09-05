# resample_binance.py: 1분봉 데이터를 다른 시간 단위로 변환(리샘플링)하는 스크립트
# 1분봉 데이터를 사용하여 5분봉, 15분봉, 1시간봉 등 더 큰 시간 단위의 데이터를 생성합니다.
# 이렇게 만들어진 데이터는 01_build_dataset.py에서 멀티타임프레임 피처를 만드는 데 사용됩니다.

import pandas as pd
import os

# --- 설정 ---
# 리샘플링할 원본 1분봉 데이터 파일
SRC_FILE = "data/BTCUSDT-1m-202401_202507.csv"
# 리샘플링할 시간 단위 목록
RULES = ["5T", "15T", "1H"] # 5분(5T), 15분(15T), 1시간(1H)

def ohlcv_resample(df, rule):
    """주어진 시간 단위(rule)에 맞춰 OHLCV 데이터를 리샘플링하는 함수"""
    # resample() 함수는 시간 인덱스를 기준으로 데이터를 그룹화합니다.
    o = df["open"].resample(rule).first()   # 기간 내 첫번째 값
    h = df["high"].resample(rule).max()     # 기간 내 최대값
    l = df["low"].resample(rule).min()      # 기간 내 최소값
    c = df["close"].resample(rule).last()    # 기간 내 마지막 값
    v = df["volume"].resample(rule).sum()    # 기간 내 합계
    
    # 계산된 O,H,L,C,V를 하나의 데이터프레임으로 합치고, 결측치가 있는 행은 제거합니다.
    out = pd.concat([o,h,l,c,v], axis=1).dropna()
    out.columns = ["open","high","low","close","volume"]
    return out

# --- 메인 로직 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    # datetime 컬럼을 파싱하고, 시간 기반으로 리샘플링하기 위해 인덱스로 설정합니다.
    df = pd.read_csv(SRC_FILE, parse_dates=["datetime"])
    df = df.set_index("datetime")

    print(f"Resampling source file: {SRC_FILE}")

    # 2. 리샘플링 및 저장
    # RULES 목록에 있는 각 시간 단위에 대해 루프를 돕니다.
    for rule in RULES:
        print(f"Resampling to {rule}...")
        resampled_df = ohlcv_resample(df, rule)
        
        # 저장할 파일 경로를 생성합니다. (예: data/BTCUSDT-5m-202401_202507.csv)
        out_path = f"data/BTCUSDT-{rule.lower()}-202401_202507.csv".replace("t","m")
        
        # 인덱스(datetime)를 다시 컬럼으로 되돌리고 CSV 파일로 저장합니다.
        resampled_df.reset_index().to_csv(out_path, index=False)
        print(f"  -> Saved to: {out_path}, Rows: {len(resampled_df)}")

    print("\nResampling complete.")