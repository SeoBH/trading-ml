# 08_promote_winner.py: A/B 테스트 승자 모델을 실제 운영에 반영하는 스크립트
# 07_score_ab.py가 결정한 승자(A 또는 B) 정보를 읽어와서,
# 앞으로 실시간 거래에 사용될 모델을 승자 모델로 교체(Promote)하는 역할을 합니다.
# 이 스크립트는 MLOps 파이프라인의 마지막 단계로, 모델 배포를 자동화합니다.

import json, sys, os

# === 경로 설정 ===
# 실시간 운영에서 어떤 모델을 사용할지 정의하는 설정 파일
ACTIVE_CONFIG_PATH = "config/active_models.json"
# A/B 테스트의 승자 정보가 기록된 파일
SCORE_FILE_PATH = "logs/ab_score.txt"

# --- 메인 로직 ---
if __name__ == "__main__":
    # 1. 승자 정보 읽기
    # ab_score.txt 파일이 없으면, A/B 테스트가 아직 실행되지 않은 것이므로 에러를 내고 종료합니다.
    if not os.path.exists(SCORE_FILE_PATH):
        print(f"[ERROR] Score file not found at: {SCORE_FILE_PATH}")
        print("Please run A/B test and scoring first.")
        sys.exit(1)

    # 파일에서 승자 정보('A' 또는 'B')를 읽어옵니다.
    with open(SCORE_FILE_PATH) as f: 
        winner = f.read().strip()

    if winner not in ['A', 'B']:
        print(f"[ERROR] Invalid winner specified in score file: {winner}")
        sys.exit(1)

    print(f"The winner is Model {winner}. Promoting it to production...")

    # 2. 새로운 운영 설정 생성
    # 승자에 따라 운영에 사용할 모델 파일 경로를 결정합니다.
    if winner == "A":
        new_config = {
            "model_reg": "models/lgbm_retH_A.pkl",
            "model_cls": "models/lgbm_cls_A.pkl"
        }
    else: # winner == "B"
        new_config = {
            "model_reg": "models/lgbm_retH_B.pkl",
            "model_cls": "models/lgbm_cls_B.pkl"
        }

    # 3. 설정 파일 업데이트
    # config 폴더가 없으면 생성합니다.
    os.makedirs("config", exist_ok=True)
    
    # 결정된 모델 경로가 담긴 new_config 내용을 active_models.json 파일에 덮어씁니다.
    # JSON 형식으로 이쁘게(indent=2) 저장합니다.
    with open(ACTIVE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)

    print(f"[SUCCESS] {ACTIVE_CONFIG_PATH} has been updated.")
    print(f"Model {winner} is now the active production model.")