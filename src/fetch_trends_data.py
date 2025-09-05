# fetch_trends_data.py: Google Trends 데이터를 수집하는 스크립트
# (3개월 단위 요청, 타임아웃, 진행률 표시 기능 추가 버전)

import pandas as pd
from pytrends.request import TrendReq
import time
import os

# --- 설정 ---
KEYWORDS = ['Bitcoin'] # 트렌드를 가져올 키워드
# freq='3MS'는 3개월 단위로 날짜를 생성합니다.
DATE_RANGES = pd.date_range(start='2024-01-01', end='2025-07-31', freq='3MS')
OUT_FILE = 'data/google_trends.csv' # 저장할 파일 경로

def fetch_trends(pytrends, keyword, timeframe):
    """특정 기간의 시간별 트렌드 데이터를 가져옵니다."""
    try:
        # build_payload: 어떤 키워드와 기간의 데이터를 가져올지 구글에 요청을 준비하는 단계
        pytrends.build_payload(kw_list=[keyword], timeframe=timeframe, geo='', gprop='')
        # interest_over_time: 준비된 요청을 바탕으로 실제 데이터를 가져오는 단계
        df = pytrends.interest_over_time()
        return df
    except Exception as e:
        # 429 에러는 요청 속도 제한을 의미합니다.
        if '429' in str(e):
            print(f"    -> Rate limit hit. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            return fetch_trends(pytrends, keyword, timeframe) # 60초 후 재귀 호출로 재시도
        else:
            print(f"    -> An error occurred: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # TrendReq: pytrends 라이브러리를 사용하기 위한 메인 객체 생성
    # timeout=(10, 25) -> 연결 시도 10초, 연결 후 응답 25초를 초과하면 에러 발생
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
    
    print(f"Fetching Google Trends data for keywords: {KEYWORDS}")
    
    all_trends_df = pd.DataFrame()
    # 2025-07-31을 포함시키기 위해 마지막 날짜를 명시적으로 추가
    request_dates = DATE_RANGES.to_list()
    if request_dates[-1].strftime('%Y-%m-%d') < '2025-07-31':
        request_dates.append(pd.to_datetime('2025-08-01'))

    total_requests = len(request_dates) - 1

    # date_ranges를 순회하며 3개월 단위로 데이터 요청
    for i in range(total_requests):
        start_date = request_dates[i]
        end_date = request_dates[i+1] - pd.Timedelta(days=1)

        timeframe_str = f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}'
        
        print(f"\n[PROGRESS {i+1}/{total_requests}] Fetching data for {timeframe_str}...")
        
        for keyword in KEYWORDS:
            trends_df = fetch_trends(pytrends, keyword, timeframe_str)
            if not trends_df.empty:
                # isPartial 컬럼은 불필요하므로 제거
                if 'isPartial' in trends_df.columns:
                    trends_df = trends_df.drop(columns=['isPartial'])
                all_trends_df = pd.concat([all_trends_df, trends_df])
            # 요청 간에 약간의 딜레이를 주어 차단을 방지
            time.sleep(5)

    if not all_trends_df.empty:
        # 인덱스(날짜)를 UTC 시간대로 통일하여 다른 데이터와 쉽게 병합하도록 합니다.
        all_trends_df.index = all_trends_df.index.tz_localize('UTC')
        # 컬럼 이름을 나중에 알아보기 쉽게 'trends_score'로 변경합니다.
        all_trends_df.rename(columns={'Bitcoin': 'trends_score'}, inplace=True)
        
        os.makedirs('data', exist_ok=True)
        all_trends_df.to_csv(OUT_FILE)
        print(f"\n[SUCCESS] Successfully fetched and saved data to {OUT_FILE}")
        print(f"Total rows: {len(all_trends_df)}")
    else:
        print("\n[FAILED] Failed to fetch any data.")