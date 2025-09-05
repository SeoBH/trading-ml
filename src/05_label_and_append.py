# 05_label_and_append.py: 실시간 로그에 정답을 붙여 누적 저장하는 스크립트
# 실시간 운영(04_live_infer_ws.py)을 통해 생성된 로그 파일에는 "예측"만 있고 "정답"은 없습니다.
# 이 스크립트는 시간이 흐른 뒤, 그 예측이 맞았는지 틀렸는지 "정답(미래 수익률)"을 계산하여
# 로그 데이터와 결합한 뒤, 영구적인 학습 저장소에 추가하는 역할을 합니다.

import pandas as pd
from pathlib import Path
import os

# === 설정 ===
H = 60  # 정답(레이블)을 계산하기 위한 시간 (60분 뒤 수익률)

# 사용할 파일 경로
RAW_PRICES = "data/BTCUSDT-1m-202401_202507.csv"  # 원본 가격 데이터
# --- FIX: 실전 투자 로그도 학습에 포함하도록 경로 추가 ---
REPLAY_LOG = "logs/replay/replay_full.csv"
PROD_LOG = "logs/prod/prod.csv"
# ----------------------------------------------------
# 정답이 추가된 데이터를 영구적으로 보관할 학습 저장소
STORE = "data/training_store.parquet"

# --- 메인 로직 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    # 원본 가격 데이터와, 예측 로그를 불러옵니다.
    raw = pd.read_csv(RAW_PRICES, parse_dates=["datetime"]).set_index("datetime").sort_index()
    
    # --- FIX: 리플레이 로그와 실전 투자 로그를 모두 불러와서 합칩니다. ---
    print(f"Loading logs...")
    log_files = []
    if os.path.exists(REPLAY_LOG):
        print(f" - Found replay log: {REPLAY_LOG}")
        log_files.append(pd.read_csv(REPLAY_LOG, parse_dates=["datetime"]))
    
    if os.path.exists(PROD_LOG):
        print(f" - Found production log: {PROD_LOG}")
        log_files.append(pd.read_csv(PROD_LOG, parse_dates=["datetime"]))

    if not log_files:
        print("[ERROR] No log files found. Exiting.")
        exit()
        
    pred_raw = pd.concat(log_files)
    print(f"Total {len(pred_raw)} log entries loaded.")
    
    # 중복된 로그가 있을 수 있으므로, datetime 기준으로 중복을 제거하고 정렬합니다.
    pred = pred_raw.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
    print(f"Processing {len(pred)} unique log entries...")
    # --------------------------------------------------------------------

    # 2. 정답(레이블) 계산
    # 예측 로그의 각 행(예측 시점)에 대해, H분 뒤의 실제 가격이 얼마였는지를 찾아냅니다.
    
    # 예측 시점의 종가를 로그 데이터에 결합
    df = pred.join(raw[["close"]].rename(columns={"close":"close_now"}), how="left")
    
    # H분 뒤의 종가를 원본 가격 데이터에서 찾아와 결합
    # shift(-H)는 데이터를 H행만큼 위로 당겨, 현재 행에 미래 가격을 붙이는 효과입니다.
    df["close_future"] = raw["close"].shift(-H).reindex(df.index)
    
    # H분 뒤 수익률 (이것이 바로 우리가 예측했던 것에 대한 "정답"입니다)
    df["ret_H"] = df["close_future"]/df["close_now"] - 1

    # 3. 레이블이 생긴 데이터만 필터링 및 저장 준비
    # H분 뒤의 가격을 아직 알 수 없는 최신 데이터들은 정답(ret_H)이 비어있게(NaN) 됩니다.
    # 여기서는 정답이 있는 행만 골라냅니다.
    labeled = df.dropna(subset=["ret_H"]).reset_index()

    # 4. 누적 저장소에 추가 (Append)
    Path("data").mkdir(exist_ok=True) # data 폴더가 없으면 생성
    try:
        # 기존 학습 저장소가 있으면, 불러와서 새로운 데이터를 이어붙입니다.
        store = pd.read_parquet(STORE)
        # 중복된 데이터가 없도록 datetime을 기준으로 중복을 제거하고 합칩니다.
        out = pd.concat([store, labeled]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    except FileNotFoundError:
        # 기존 저장소가 없으면, 지금 만든 데이터가 첫 데이터가 됩니다.
        out = labeled.sort_values("datetime")

    # 최종 결과를 Parquet 형식으로 저장합니다.
    # Parquet은 CSV보다 용량이 작고 읽기/쓰기 속도가 빨라 대용량 데이터에 유리합니다.
    out.to_parquet(STORE, index=False)
    
    print(f"Labeled data appended: {len(labeled)} rows")
    print(f"Total rows in training store: {len(out)}")