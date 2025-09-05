# 02_train_eval.py: 모델 학습 및 평가 스크립트
# (Google Trends 피처 포함 데이터셋용)

import os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from lightgbm import LGBMRegressor, LGBMClassifier
from joblib import dump

warnings.filterwarnings("ignore", category=UserWarning)

# === 경로 및 설정값 수정 ===
DATA_PATH = "data/dataset_1m_H60_with_trends.csv"  # <--- FIX: 새로운 데이터셋 경로
MODEL_DIR = "models"
REG_OUT   = os.path.join(MODEL_DIR,"lgbm_retH_trends.pkl") # <--- FIX: 새로운 모델 이름
CLS_OUT   = os.path.join(MODEL_DIR,"lgbm_cls_trends.pkl")   # <--- FIX: 새로운 모델 이름
FEAT_JSON = os.path.join(MODEL_DIR,"feature_names_trends.json") # <--- FIX: 새로운 피처 목록 이름

TARGET_REG = "ret_H"
TARGET_CLS = "y_tp_sl"
VALID_RATIO = 0.2
RANDOM_SEED = 42

def load_dataset():
    return pd.read_csv(DATA_PATH, parse_dates=['datetime'])

def infer_feats(df):
    drop = [TARGET_REG, TARGET_CLS, "y_cls", "datetime", "time", "date", "symbol"]
    return [c for c in df.columns if c not in drop and df[c].dtype != object and not c.startswith('fut_')]

def split(df):
    n=len(df); n_val=int(n*VALID_RATIO)
    return np.arange(0,n-n_val), np.arange(n-n_val,n)

def train_reg(X,y,Xv,yv):
    m=LGBMRegressor(n_estimators=5000,learning_rate=0.03,num_leaves=64,random_state=RANDOM_SEED)
    m.fit(X,y,eval_set=[(Xv,yv)],eval_metric="l2", callbacks=[])
    return m

def train_cls(X,y,Xv,yv):
    pos=(y==1).sum(); neg=(y==0).sum()
    spw=neg/max(pos,1)
    m=LGBMClassifier(n_estimators=4000,learning_rate=0.03,num_leaves=64,
                     scale_pos_weight=spw,random_state=RANDOM_SEED)
    m.fit(X,y,eval_set=[(Xv,yv)],eval_metric="auc", callbacks=[])
    return m

def main():
    df=load_dataset()
    
    if TARGET_CLS not in df:
        raise ValueError(f"분류 타겟 '{TARGET_CLS}' 컬럼이 데이터셋에 없습니다.")
    if TARGET_REG not in df:
        raise ValueError(f"회귀 타겟 '{TARGET_REG}' 컬럼이 데이터셋에 없습니다.")

    feats=infer_feats(df)
    df=df.dropna(subset=feats+[TARGET_REG,TARGET_CLS])
    tr,va=split(df)
    X,y_reg,y_cls=df[feats],df[TARGET_REG],df[TARGET_CLS]
    
    print(f"Training with {len(feats)} features (including trends_score)...")

    reg_model=train_reg(X.iloc[tr],y_reg.iloc[tr],X.iloc[va],y_reg.iloc[va])
    cls_model=train_cls(X.iloc[tr],y_cls.iloc[tr],X.iloc[va],y_cls.iloc[va])
    
    Path(MODEL_DIR).mkdir(exist_ok=True)
    dump(reg_model,REG_OUT)
    dump(cls_model,CLS_OUT)
    
    fnames=list(reg_model.booster_.feature_name())
    with open(FEAT_JSON,"w",encoding="utf-8") as f: 
        json.dump(fnames,f,indent=2)
        
    print(f"[SUCCESS] New models saved: {REG_OUT}, {CLS_OUT}")

if __name__=="__main__": 
    main()