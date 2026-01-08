import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. 파일 경로 설정 (사용자분이 알려주신 경로 기준)
base_path = r'C:\Users\USER\Desktop\AI 데이터 분석 4\workspace\project'
csv_path = os.path.join(base_path, '비만_등급_SMOTE.csv')
model_save_path = os.path.join(base_path, 'obesity_model.pkl')

print(f"📂 데이터 읽는 중: {csv_path}")

# 2. 데이터 로드 및 학습
try:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['NObeyesdad'])
    y = df['NObeyesdad']
    
    # 모델 학습 (Random Forest)
    print("🤖 AI 모델 학습 시작... (잠시만 기다려주세요)")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # 3. 모델 저장 (.pkl 파일 생성)
    joblib.dump(rf_model, model_save_path)
    print("-" * 50)
    print("✅ 성공! 모델 파일이 아래 위치에 저장되었습니다:")
    print(f"👉 {model_save_path}")
    print("-" * 50)
    print("이제 터미널에 'streamlit run app.py'를 입력하면 앱이 실행됩니다!")

except FileNotFoundError:
    print(f"❌ 에러: '{csv_path}' 파일을 찾을 수 없습니다.")
    print("폴더 안에 '비만_등급_SMOTE.csv' 파일이 있는지 확인해주세요.")