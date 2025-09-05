# 06_retrain_from_store.py: 누적 데이터로 모델을 재학습하는 스크립트
# 실시간 운영(04_live_infer_ws.py)을 통해 얻은 새로운 데이터에 정답(레이블)을 붙여
# 누적 저장소(training_store.parquet)에 저장하고, 이 데이터를 이용해 모델을 주기적으로 재학습합니다.
# 이를 통해 모델은 최신 시장 상황을 계속 학습하여 성능을 유지하거나 개선할 수 있습니다.

import pandas as pd, numpy as np, os, time
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from joblib import dump
# 01_build_dataset.py와 동일한 피처 계산을 위해 라이브러리를 가져옵니다.
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

# === 경로 및 설정값 ===
STORE = "data/training_store.parquet" # 누적 학습 데이터 저장소 경로
MODEL_DIR = "models"                  # 재학습된 모델을 저장할 폴더
os.makedirs(MODEL_DIR, exist_ok=True)

# 재학습에 사용할 데이터의 최대 행 수. 너무 오래된 데이터는 제외하여 최신 경향에 집중합니다.
MAX_ROWS = 200_000

# --- 함수 정의 ---
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 OHLCV 데이터프레임에서 피처를 계산합니다.
    이 함수는 01_build_dataset.py의 피처 생성 로직과 반드시 동일해야 합니다.
    학습, 백테스트, 실시간 추론, 재학습 모두에서 동일한 '세상'을 보도록 보장하는 핵심적인 부분입니다.
    """
    # parquet 파일에서 읽은 데이터는 datetime 인덱스가 없으므로 다시 설정해줍니다.
    df = df.set_index('datetime').sort_index()

    # --- 아래는 01_build_dataset.py와 동일한 피처 계산 로직 --- #
    df['ret1'] = df['close'].pct_change(1)
    for w in [3, 5, 15, 30, 60]:
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
    
    return df

# --- 메인 로직 ---
# 1. 누적 데이터 저장소에서 데이터 로드
df_raw = pd.read_parquet(STORE).sort_values("datetime")
# 너무 오래된 데이터는 잘라냅니다.
if len(df_raw) > MAX_ROWS:
    df_raw = df_raw.iloc[-MAX_ROWS:]

# 2. 피처 재계산
# 저장소에는 OHLCV 데이터와 함께 과거 예측 로그 등 다른 정보도 섞여 있습니다.
# 이 중에서 순수한 시장 데이터(OHLCV)와 정답(ret_H)만 골라내어 피처를 다시 계산해야 합니다.
cols_for_feature_gen = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'ret_H']
# 만약 저장소에 OHLCV 일부가 없다면, close 값으로 채워넣는 등 예외처리를 합니다.
if 'open' not in df_raw.columns: df_raw['open'] = df_raw['close']
if 'high' not in df_raw.columns: df_raw['high'] = df_raw['close']
if 'low' not in df_raw.columns: df_raw['low'] = df_raw['close']
if 'volume' not in df_raw.columns: df_raw['volume'] = 0

# 핵심: 순수 데이터로 피처를 다시 계산합니다.
df_with_feats = compute_features(df_raw[cols_for_feature_gen].copy())

# 3. 학습 데이터 준비
label_col = "ret_H"
# 학습에 사용하면 안되는 컬럼들(원본 OHLCV, 로그 기록 등)을 제외하고 순수 피처만 선택합니다.
base_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'pv', 'cum_v', 'cum_pv']
log_artefacts = [c for c in df_raw.columns if c.startswith(('pred_', 'proba', 'model_', 'tag', 'action', 'equity'))]
feat_cols = [c for c in df_with_feats.columns if c not in base_cols + [label_col] + log_artefacts]

# 결측치가 있는 행을 제거하여 최종 학습 데이터를 완성합니다.
df_final = df_with_feats.dropna(subset=feat_cols + [label_col])
X = df_final[feat_cols]
y = df_final[label_col]

# 4. 회귀 모델 재학습
# 여기서는 회귀 모델만 재학습하는 예시를 보여줍니다. (분류 모델도 동일하게 가능)
reg = LGBMRegressor(n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=0)
# TimeSeriesSplit은 시계열 데이터용 교차 검증기. 여기서는 재학습 과정의 성능을 참고용으로 출력하는 데 사용됩니다.
tscv = TimeSeriesSplit(n_splits=3)

print(f"{len(X)}개의 샘플로 모델 재학습을 시작합니다...")
for i, (tr, te) in enumerate(tscv.split(X), 1):
    reg.fit(X.iloc[tr], y.iloc[tr])
    pred = reg.predict(X.iloc[te])
    mse = mean_squared_error(y.iloc[te], pred)
    print(f"[재학습][Fold{i}] MSE={mse:.6e}")

# 5. 재학습된 모델 저장
# 파일 이름에 현재 시간을 태그로 붙여, 어떤 모델이 언제 학습되었는지 구별할 수 있게 합니다.
tag = time.strftime("%Y%m%d%H")
out_path = os.path.join(MODEL_DIR, f"lgbm_retH_{tag}.pkl")
dump(reg, out_path)
print("저장 완료:", out_path)
