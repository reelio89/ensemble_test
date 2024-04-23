# 0.
# pip install lightgbm==3.3.5

# 1. 데이터 준비
# 데이터 준비 단계는 분류 문제와 동일하게, 특성(features)과 타겟 변수(target)를 준비하고, LightGBM의 데이터셋 형식으로 변환합니다.

## CODE
from sklearn import datasets
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd

# California Housing 데이터셋 로드
california_data = datasets.fetch_california_housing()
X = pd.DataFrame(california_data.data, columns=california_data.feature_names)
y = pd.Series(california_data.target)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 데이터셋 생성
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


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

# 모델 학습을 위한 파라미터 설정
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 모델 학습
gbm = lgb.train(params,
                train_data,
                num_boost_round=100,
                valid_sets=test_data,
                early_stopping_rounds=10)


# 3. 예측 및 평가
# 회귀 문제에서도 예측과 평가는 분류 문제와 유사한 절차를 따르며, 평가 지표만 회귀 분석에 적합한 것으로 변경합니다. 
# 예를 들어, RMSE(평균 제곱근 오차)나 MAE(평균 절대 오차) 등을 사용할 수 있습니다.

## CODE
# 예측
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

from sklearn.metrics import mean_squared_error
import numpy as np

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")

# 시각화

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import pandas as pd

# 데이터 로드
california_data = datasets.fetch_california_housing()
X = pd.DataFrame(california_data.data, columns=california_data.feature_names)
y = pd.Series(california_data.target, name='MedianHouseValue')

# 데이터 프레임에 타겟 변수 추가
data = pd.concat([X, y], axis=1)

# 소득 대비 주택 가격
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MedInc', y='MedianHouseValue', data=data, alpha=0.2)
plt.title('Median Income vs. Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.show()

# 주택 연령 대비 주택 가격
plt.figure(figsize=(10, 6))
sns.scatterplot(x='HouseAge', y='MedianHouseValue', data=data, alpha=0.2)
plt.title('House Age vs. Median House Value')
plt.xlabel('House Age')
plt.ylabel('Median House Value')
plt.show()

# 지리적 위치와 주택 가격
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='MedianHouseValue', size='Population',
                sizes=(10, 200), alpha=0.5, palette='coolwarm', data=data)
plt.title('Geographical Distribution of Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
