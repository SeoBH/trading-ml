# 01_build_dataset.py: 학습 데이터셋 생성 스크립트
# (Google Trends 피처 추가 버전)

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

# === 경로 설정 ===
SRC_PRICE = "data/BTCUSDT-1m-202401_202507.csv"
SRC_TRENDS = "data/google_trends.csv" # 트렌드 데이터 경로 추가
OUT = "data/dataset_1m_H60_with_trends.csv" # 결과 파일 이름 변경

# === 하이퍼파라미터 ===
H = 60

def resample_ohlcv(df, rule: str):
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = [f'open_{rule}', f'high_{rule}', f'low_{rule}', f'close_{rule}', f'vol_{rule}']
    return out

if __name__ == "__main__":
    # 1. 데이터 로드 및 기본 전처리
    df = pd.read_csv(SRC_PRICE, parse_dates=['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    # --- Google Trends 데이터 로드 및 병합 --- #
    try:
        trends_df = pd.read_csv(SRC_TRENDS, parse_dates=['date'], index_col='date')
        # 1분봉 데이터프레임에 트렌드 데이터를 병합 (left join)
        df = pd.merge(df, trends_df, left_index=True, right_index=True, how='left')
        # 트렌드 데이터는 시간별이므로, 비어있는 분(minute)은 직전 시간의 값으로 채웁니다 (forward-fill).
        df['trends_score'].ffill(inplace=True)
        # 혹시 맨 처음에 값이 비어있을 경우를 대비해 0으로 채웁니다.
        df['trends_score'].fillna(0, inplace=True)
        print("Google Trends data successfully merged.")
    except FileNotFoundError:
        print(f"Warning: {SRC_TRENDS} not found. Skipping trends feature.")
        df['trends_score'] = 0 # 트렌드 파일이 없으면 0으로 채움
    # ----------------------------------------- #

    # 2. 피처 엔지니어링
    df['ret1'] = df['close'].pct_change(1)
    for w in [3,5,15,30,60]:
        df[f'ret{w}'] = df['close'].pct_change(w)
        df[f'vol{w}'] = df['close'].pct_change().rolling(w).std()
        df[f'roll_mean{w}'] = df['close'].rolling(w).mean()
        df[f'roll_max{w}'] = df['high'].rolling(w).max()
        df[f'roll_min{w}'] = df['low'].rolling(w).min()

    df['rsi14'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['atr14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    df['pv'] = df['close'] * df['volume']
    df['cum_v'] = df['volume'].groupby(df.index.date).cumsum()
    df['cum_pv'] = df['pv'].groupby(df.index.date).cumsum()
    df['vwap_day'] = df['cum_pv'] / df['cum_v']

    df_5m = resample_ohlcv(df[['open','high','low','close','volume']], '5T')
    df_15m = resample_ohlcv(df[['open','high','low','close','volume']], '15T')
    df_mt = pd.concat([df_5m, df_15m], axis=1).reindex(df.index).ffill()
    df = pd.concat([df, df_mt], axis=1)

    # 3. 레이블링
    df['ret_H'] = df['close'].shift(-H) / df['close'] - 1
    TP = 0.0015
    SL = 0.0010
    df['fut_high_max'] = df['high'].shift(-1).rolling(H).max()
    df['fut_low_min']  = df['low'].shift(-1).rolling(H).min()
    hit_tp = (df['fut_high_max'] >= df['close'] * (1+TP)).astype(int)
    hit_sl = (df['fut_low_min']  <= df['close'] * (1-SL)).astype(int)
    df['y_tp_sl'] = np.where(hit_sl==1, 0, np.where(hit_tp==1, 1, np.nan))

    # 4. 최종 데이터셋 정리 및 저장
    cols_keep = [
        'open','high','low','close','volume','ret1','ret3','ret5','ret15','ret30','ret60',
        'vol3','vol5','vol15','vol30','vol60','rsi14','macd','macd_signal','macd_diff','atr14',
        'vwap_day',
        'open_5T','high_5T','low_5T','close_5T','vol_5T',
        'open_15T','high_15T','low_15T','close_15T','vol_15T',
        'trends_score'  # --- 새로운 피처 추가 ---
    ]

    out_df = df[cols_keep + ['ret_H','y_tp_sl']].dropna()
    
    out_df.to_csv(OUT)
    print(f"피처/레이블 생성 완료. 저장 경로: {OUT}, 총 행 수: {len(out_df)}")
