# 0.
# pip install lightgbm

# 1. 데이터 준비
## CODE

import pandas as pd
import lightgbm as lgb

# 데이터 로드 (예: train.csv, test.csv)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 특성과 타겟 변수 분리
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# LightGBM 데이터셋 생성
train_dataset = lgb.Dataset(X_train, label=y_train)
test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)

# 2. 모델 학습
# LightGBM 모델을 학습시키기 위해 파라미터를 설정하고, train 함수를 사용합니다. 
# 파라미터는 학습률, 최대 깊이, 트리의 개수 등 학습 과정과 모델 성능에 영향을 주는 요소들을 조정합니다.

## CODE
params = {
    'boosting_type': 'gbdt', # Gradient Boosting Decision Tree
    'objective': 'binary', # 이진 분류
    'metric': 'binary_logloss', # 평가 지표
    'num_leaves': 31, # 한 트리가 가질 수 있는 최대 리프 수
    'learning_rate': 0.05, # 학습률
    'feature_fraction': 0.9, # 트리를 생성할 때마다 선택할 피처의 비율
    'bagging_fraction': 0.8, # 데이터를 샘플링하는 비율
    'bagging_freq': 5 # k회 반복마다 배깅을 수행
}

# 모델 학습
gbm = lgb.train(params,
                train_dataset,
                num_boost_round=100, # 부스팅 반복 횟수
                valid_sets=test_dataset, # 평가 데이터셋
                early_stopping_rounds=10) # 조기 종료 조건

# 4. 예측 및 평가

## CODE
# 예측
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 성능 평가 (예: AUC 점수 계산)
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test, y_pred)
print(f'Test AUC score: {auc_score}')

# LightGBM은 매우 유연하고 다양한 파라미터를 제공하기 때문에, 
# 사용자의 특정 목적에 맞게 파라미터를 조정할 필요가 있습니다. 
# 공식 문서에서 더 많은 파라미터와 사용 방법에 대한 정보를 확인할 수 있습니다.

