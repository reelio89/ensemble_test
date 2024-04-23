# 0.
# pip install lightgbm==3.3.5

# 1. 데이터 준비
# 데이터 준비 단계는 분류 문제와 동일하게, 특성(features)과 타겟 변수(target)를 준비하고, LightGBM의 데이터셋 형식으로 변환합니다.

## CODE

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb

# Boston Housing Dataset을 fetch_openml을 통해 불러오기
X, y = datasets.fetch_openml('boston', version=1, return_X_y=True, as_frame=True)

# 데이터를 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 데이터셋 생성
train_dataset = lgb.Dataset(X_train, label=y_train)
test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)

# 확인을 위한 코드 (선택적)
print("훈련 세트 크기:", X_train.shape)
print("테스트 세트 크기:", X_test.shape)


# 2. 모델 학습

# metric='12' : l2 distance(MSE)를 의미
# feature_fraction : 트리를 훈련시킬 때마다 무작위로 선택되는 피처의 비율
## feature_fraction=0.9 -> 각 반복에서 사용되는 피처의 90%만을 무작위로 선택 및 학습
# bagging_fraction  : 데이터의 일부분만을 사용하여 트리를 훈련시키는 데 사용
## bagging_fraction=0.8 -> 전체 데이터 세트의 80%를 무작위로 선택하여 트리 학습에 사용
# bagging_freq는 bagging을 수행할 빈도
## bagging_freq는=5 -> 5번의 부스팅 반복마다 새로운 데이터의 부분 집합이 선택 및 학습



## CODE

params = {
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

gbm = lgb.train(params,
                train_dataset,
                num_boost_round=200,
                valid_sets=test_dataset,
                early_stopping_rounds=20)

# 3. 예측 및 평가
# 회귀 문제에서도 예측과 평가는 분류 문제와 유사한 절차를 따르며, 평가 지표만 회귀 분석에 적합한 것으로 변경합니다. 
# 예를 들어, RMSE(평균 제곱근 오차)나 MAE(평균 절대 오차) 등을 사용할 수 있습니다.

## CODE
# 예측
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 성능 평가 (예: RMSE 계산)
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')

# 위의 예시처럼, LightGBM을 회귀 분석에 사용할 때는 주로 목표 변수 설정과 평가 지표 선택에서 차이가 발생합니다.
# 나머지 부분은 분류 문제를 해결할 때와 매우 유사합니다. 
# LightGBM은 회귀와 분류 모두에서 높은 성능을 발휘할 수 있는 강력한 알고리즘입니다.