# merge_binance_1m.py: 여러 개의 바이낸스 1분봉 CSV 파일을 하나로 병합하는 스크립트
# 바이낸스에서 월별로 내려받은 데이터들을 하나의 파일로 합쳐, 데이터 전처리를 쉽게 만듭니다.

import pandas as pd
import glob, os
import numpy as np

# --- 설정 ---
# 병합할 파일들이 있는 폴더 경로
SRC_DIR = "data/binance/"
# 병합된 결과 파일을 저장할 경로
OUT_FILE = "data/BTCUSDT-1m-202401_202507.csv"
# 찾을 파일 패턴 (2024년, 2025년 1월~7월의 BTCUSDT 1분봉 데이터)
FILE_PATTERN = "BTCUSDT-1m-202[4-5]*.csv"

def parse_datetime_col(df):
    """
    Binance 덤프의 open_time 컬럼을 안전하게 datetime으로 변환합니다.
    - 정수형이면 자리수로 단위(s, ms, us, ns)를 자동 감지합니다.
    - 문자열이면 그대로 to_datetime으로 파싱합니다.
    """
    col = df["open_time"]

    if np.issubdtype(col.dtype, np.number):
        vals = col.astype("int64", copy=False)
        maxv = int(vals.max())
        digits = len(str(maxv))

        if digits <= 11: unit = "s"      # 초 단위 (e.g., 1700000000)
        elif digits <= 14: unit = "ms"   # 밀리초 단위 (e.g., 1700000000000)
        else: unit = "us"              # 마이크로초 이상
        
        return pd.to_datetime(vals, unit=unit, utc=True)
    else:
        # 숫자 형태가 아닌 ISO 날짜 문자열 등
        return pd.to_datetime(col, utc=True, errors="raise")

# --- 메인 로직 ---
if __name__ == "__main__":
    # 1. 파일 목록 가져오기
    # glob을 사용하여 패턴에 맞는 모든 파일의 경로를 리스트로 가져옵니다.
    files = sorted(glob.glob(os.path.join(SRC_DIR, FILE_PATTERN)))
    if not files:
        print(f"[ERROR] No CSV files found in {SRC_DIR} matching the pattern.")
        exit()

    print(f"Found {len(files)} files to merge.")

    # 2. 파일 병합
    dfs = [] # 각 파일에서 읽어온 데이터프레임을 저장할 리스트
    for f in files:
        print(f"Processing {f}...")
        # 파일에 헤더(컬럼 이름)가 없는 경우가 많으므로, header=None으로 읽고 직접 컬럼 이름을 지정합니다.
        df = pd.read_csv(f, header=None, low_memory=False)
        df = df.iloc[:, :12] # 필요한 12개 컬럼만 선택
        df.columns = [
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ]
        
        # open_time을 표준 datetime 형식으로 변환
        df["datetime"] = parse_datetime_col(df)

        # 필요한 컬럼만 선택하고 데이터 타입을 숫자로 변환
        df = df[["datetime","open","high","low","close","volume"]].astype({
            "open":float,"high":float,"low":float,"close":float,"volume":float
        })
        dfs.append(df)

    # 3. 최종 데이터프레임 생성 및 후처리
    # 모든 데이터프레임을 하나로 합칩니다.
    big_df = pd.concat(dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    
    # 중복된 시간의 데이터가 있을 경우, 하나만 남기고 제거합니다.
    before_dedup = len(big_df)
    big_df = big_df.drop_duplicates(subset=["datetime"])
    print(f"Removed {before_dedup - len(big_df)} duplicate rows.")

    # 데이터가 빠진 곳(gap)이 있는지 확인합니다.
    big_df["gap_min"] = big_df["datetime"].diff().dt.total_seconds().div(60)
    gaps = big_df[big_df["gap_min"] > 1]
    if not gaps.empty:
        print("\n[WARNING] Gaps found in data:")
        print(gaps)
    else:
        print("\n[SUCCESS] No gaps found in data.")

    # 4. 결과 저장
    os.makedirs("data", exist_ok=True)
    big_df[["datetime","open","high","low","close","volume"]].to_csv(OUT_FILE, index=False)

    print(f"\nMerge complete. Saved to: {OUT_FILE}")
    print(f"Total rows: {len(big_df)}")
