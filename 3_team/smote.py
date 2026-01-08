import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# 1. 데이터 로드
df = pd.read_csv('C:/Users/USER/Desktop/AI 데이터 분석 4/workspace/project/비만 등급_전처리_최종.csv', encoding='utf-8-sig')
print("원본 데이터 로드 완료")
print(f"데이터 크기: {df.shape}")
print()

# 2. X, y 분리
y = df['NObeyesdad']
X = df.drop('NObeyesdad', axis=1)

# 3. 특성 제거: Height, Weight 컬럼 삭제
X = X.drop(['Height', 'Weight'], axis=1)
print("특성 변수 제거 완료 (Height, Weight 삭제)")
print()

# 5. SMOTE 적용 전 클래스 개수 확인
print("=" * 50)
print("SMOTE 적용 전 NObeyesdad 클래스별 개수")
print("=" * 50)
print(y.value_counts().sort_index())
print()

# 4. SMOTE 적용
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# 5. SMOTE 적용 후 클래스 개수 확인
print("=" * 50)
print("SMOTE 적용 후 NObeyesdad 클래스별 개수")
print("=" * 50)
print(pd.Series(y_smote).value_counts().sort_index())
print()

# 6. 결과 저장
# 늘어난 데이터를 하나의 DataFrame으로 합치기
X_smote_df = pd.DataFrame(X_smote, columns=X.columns)
y_smote_df = pd.Series(y_smote, name='NObeyesdad')
result_df = pd.concat([X_smote_df, y_smote_df], axis=1)

# CSV로 저장
result_df.to_csv('비만_등급_SMOTE.csv', index=False, encoding='utf-8-sig')
print("=" * 50)
print("SMOTE 처리된 데이터 저장 완료")
print("=" * 50)
print(f"저장 파일: 비만_등급_SMOTE.csv")
print(f"저장된 데이터 크기: {result_df.shape}")
