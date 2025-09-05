# 04_live_infer_ws.py: 실시간 추론 및 거래 실행 스크립트
# (진행 상황 표시 기능 추가 버전)

import os, csv, json, argparse, time, hashlib
from datetime import datetime, timezone
import pandas as pd
from joblib import load

try:
    from websocket import WebSocketApp
except ImportError:
    WebSocketApp = None

# ... (파일의 다른 부분은 이전과 동일) ...
SYMBOL = "btcusdt"
STREAM = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m" 
H = 60
FEE = 0.0003
SLIP = 0.0005
THR_REG = 0.0003
THR_CLS = 0.55
RISK = 1.0
WARMUP_BARS = 60
TP_PCT = 0.01
SL_ATR = 2.0

bars = []
reg = None
cls = None
entry = None
cash = 1.0
pos = 0.0

LOG_PATH = None
TAG = "PROD"
DEBUG = False

MODEL_REG_VER = None
MODEL_CLS_VER = None
ACTIVE_CFG = None
ACTIVE_MTIME = None
ACTIVE_MODELS = {"model_reg": None, "model_cls": None}
RELOAD_SEC = 30
_last_reload_check = 0.0
FEATURE_NAMES = None

def make_model_ver(path: str) -> str:
    if not path or not os.path.exists(path): return "NA"
    sha = hashlib.sha256(open(path,"rb").read()).hexdigest()[:8]
    return f"{os.path.basename(path)}#{sha}"

trends_data = None

def load_trends_data():
    global trends_data
    trends_path = "data/google_trends.csv"
    if os.path.exists(trends_path):
        print(f"[INFO] Loading trends data from {trends_path}")
        trends_data = pd.read_csv(trends_path, parse_dates=['date'], index_col='date')
        if getattr(trends_data.index, "tz", None) is None:
            trends_data.index = trends_data.index.tz_localize('UTC')

def load_feature_names():
    global FEATURE_NAMES
    fn = "models/feature_names_trends.json" # <-- FIX: trends 모델용 피처 목록 사용
    if os.path.exists(fn):
        print(f"[INFO] Loading feature names from {fn}")
        FEATURE_NAMES = json.load(open(fn,"r",encoding="utf-8"))

def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    missing = [f for f in FEATURE_NAMES if f not in X.columns]
    for f in missing: X[f] = 0.0
    return X.reindex(columns=FEATURE_NAMES).astype("float32")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df["ret1"] = df["close"].pct_change(1)
        for w in [3,5,15,30,60]:
            df[f"ret{w}"] = df["close"].pct_change(w)
            df[f"vol{w}"] = df["close"].pct_change().rolling(w).std()
            df[f"roll_mean{w}"] = df["close"].rolling(w).mean()
        from ta.momentum import RSIIndicator
        from ta.trend import MACD
        from ta.volatility import AverageTrueRange
        df["rsi14"] = RSIIndicator(df["close"]).rsi()
        m = MACD(df["close"])
        df["macd"] = m.macd(); df["macd_signal"] = m.macd_signal(); df["macd_diff"] = m.macd_diff()
        df["atr14"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        X = df.tail(1)
        if X.isna().any(axis=1).iloc[-1]: return pd.DataFrame()
        return X
    except Exception as e:
        if DEBUG: print(f"[ERROR] compute_features: {e}")
        return pd.DataFrame()

def append_log(dt,o,h,l,c,v,pred_ret,proba,action,equity):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = dict(datetime=dt.isoformat(),open=o,high=h,low=l,close=c,volume=v,
               pred_ret=pred_ret,proba=proba,action=action or "None",equity=equity,
               tag=TAG,model_reg_ver=MODEL_REG_VER,model_cls_ver=MODEL_CLS_VER)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

def load_models(mreg_path, mcls_path):
    global reg, cls, MODEL_REG_VER, MODEL_CLS_VER
    print(f"[INFO] Loading models: {mreg_path}, {mcls_path}")
    reg=load(mreg_path); cls=load(mcls_path)
    MODEL_REG_VER=make_model_ver(mreg_path); MODEL_CLS_VER=make_model_ver(mcls_path)

def try_hot_reload():
    global ACTIVE_MTIME, ACTIVE_MODELS, _last_reload_check
    now = time.time()
    if not ACTIVE_CFG or (now - _last_reload_check) < RELOAD_SEC: return
    _last_reload_check = now
    if not os.path.exists(ACTIVE_CFG): return
    mtime=os.path.getmtime(ACTIVE_CFG)
    if ACTIVE_MTIME and mtime<=ACTIVE_MTIME: return
    print(f"[HOTSWAP] Detected change in {ACTIVE_CFG}")
    cfg=json.load(open(ACTIVE_CFG,"r",encoding="utf-8"))
    mreg,mcls=cfg.get("model_reg"),cfg.get("model_cls")
    if not mreg or not mcls: return
    ACTIVE_MTIME=mtime
    if (mreg,mcls)!=(ACTIVE_MODELS.get("model_reg"),ACTIVE_MODELS.get("model_cls")):
        load_models(mreg,mcls); ACTIVE_MODELS={"model_reg":mreg,"model_cls":mcls}

def handle_bar(dt,o,h,l,c,v):
    global entry, cash, pos
    try_hot_reload()
    bars.append((dt,o,h,l,c,v))

    # 리플레이 시 bars 리스트가 무한정 커지는 것을 방지하여 속도 저하 해결
    if len(bars) > 200:
        del bars[:-200]

    if len(bars) < WARMUP_BARS:
        append_log(dt,o,h,l,c,v,float("nan"),float("nan"),None,cash); return
    df=pd.DataFrame(bars,columns=["datetime","open","high","low","close","volume"]).set_index("datetime")

    # --- ADD: 트렌드 피처 병합 ---
    if trends_data is not None:
        df = pd.merge_asof(df, trends_data, left_index=True, right_index=True, direction='backward')
        df['trends_score'] = df['trends_score'].fillna(0)
    else:
        df['trends_score'] = 0
    # --------------------------

    X=compute_features(df)
    if X.empty: return
    Xr, Xc=align_features(X.copy(),reg),align_features(X.copy(),cls)
    pred_ret=float(reg.predict(Xr)[0]); proba=float(cls.predict_proba(Xc)[0,1])
    signal_buy=(pred_ret>=THR_REG)or(proba>=THR_CLS)
    p=c; action=None
    if entry is None and pos==0 and signal_buy:
        px=p*(1+FEE+SLIP); qty=cash/px*RISK
        if qty>0: pos=qty; cash-=qty*px; entry=(dt,px,qty)
        action="BUY"; print(f"[BUY] {dt} at {px:.2f} (pred_ret:{pred_ret:.4f}, proba:{proba:.3f})")
    if entry:
        entry_dt,epx,qty=entry
        atr=df["atr14"].iloc[-1] if "atr14" in df else 0.0
        is_tp = p>=epx*(1+TP_PCT)
        is_sl = p<=epx-(atr*SL_ATR) if atr > 0 else False
        is_timeout = (dt-entry_dt).total_seconds()>=H*60
        if is_tp or is_sl or is_timeout:
            px=p*(1-FEE-SLIP); cash+=qty*px
            pnl=(px-epx)/epx; pos=0; entry=None
            reason = "TP" if is_tp else ("SL" if is_sl else "Timeout")
            action="SELL"; print(f"[SELL] {dt} at {px:.2f}, PnL={pnl:.2%}, Reason={reason}, Equity={cash:.4f}")
    equity=cash+(pos*p if pos>0 else 0.0)
    append_log(dt,o,h,l,c,v,pred_ret,proba,action,equity)

def on_message(ws,msg):
    k=json.loads(msg).get("k",{}); 
    if not k.get("x",False): return
    dt=datetime.fromtimestamp(k["t"]/1000,tz=timezone.utc)
    handle_bar(dt,float(k["o"]),float(k["h"]),float(k["l"]),float(k["c"]),float(k["v"]))

def on_open(ws): print("[WS] Connected to",STREAM)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_reg",default="models/lgbm_retH.pkl")
    ap.add_argument("--model_cls",default="models/lgbm_cls.pkl")
    ap.add_argument("--tag",default="PROD")
    ap.add_argument("--log",default="logs/prod/prod.csv")
    ap.add_argument("--active_cfg",default="config/active_models.json")
    ap.add_argument("--debug",action="store_true")
    ap.add_argument("--replay",default=None)
    # --- replay_limit 옵션 추가 ---
    ap.add_argument("--replay_limit",type=int,default=0)
    a=ap.parse_args()
    
    TAG=a.tag; LOG_PATH=a.log; ACTIVE_CFG=a.active_cfg; DEBUG=a.debug
    load_feature_names()
    load_trends_data() # <-- ADD: 트렌드 데이터 로드
    load_models(a.model_reg,a.model_cls)
    
    if a.replay:
        print(f"[REPLAY] Running replay from {a.replay}...")
        df=pd.read_csv(a.replay,parse_dates=["datetime"]).sort_values("datetime")
        total_rows = len(df)
        
        if a.replay_limit > 0:
            print(f"[REPLAY] Applying limit: processing first {a.replay_limit} of {total_rows} rows.")
            df=df.head(a.replay_limit)
        
        # --- 진행 상황 표시를 위한 변수 ---
        processed_rows = 0
        start_time = time.time()
        
        # --- 메인 루프 수정 ---
        for _,r in df.iterrows():
            dt=r["datetime"]; 
            if getattr(dt,"tzinfo",None)is None: dt=dt.tz_localize("UTC")
            handle_bar(dt,float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]),float(r["volume"]))
            
            processed_rows += 1
            # --- 5000행마다 진행 상황 출력 ---
            if processed_rows % 5000 == 0:
                elapsed = time.time() - start_time
                rows_per_sec = processed_rows / elapsed if elapsed > 0 else 0
                print(f"[PROGRESS] Processed {processed_rows} / {len(df)} rows... ({rows_per_sec:.0f} rows/sec)")

    else:
        if not WebSocketApp: raise ImportError("Please install websocket-client")
        ws=WebSocketApp(STREAM,on_open=on_open,on_message=on_message)
        ws.run_forever()