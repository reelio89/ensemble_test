# 0.
# pip install lightgbm==3.3.5

# 1. 데이터 준비
# 데이터 준비 단계는 분류 문제와 동일하게, 특성(features)과 타겟 변수(target)를 준비하고, LightGBM의 데이터셋 형식으로 변환합니다.

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

# metric='multi_logloss' : Multiclass logarithmic loss
## Multiclass logarithmic loss은 다중 클래스 분류 문제에 적용된 cross entropy 손실의 한 형태
# feature_fraction : 트리를 훈련시킬 때마다 무작위로 선택되는 피처의 비율
## feature_fraction=0.9 -> 각 반복에서 사용되는 피처의 90%만을 무작위로 선택 및 학습
# bagging_fraction  : 데이터의 일부분만을 사용하여 트리를 훈련시키는 데 사용
## bagging_fraction=0.8 -> 전체 데이터 세트의 80%를 무작위로 선택하여 트리 학습에 사용
# bagging_freq는 bagging을 수행할 빈도
## bagging_freq는=5 -> 5번의 부스팅 반복마다 새로운 데이터의 부분 집합이 선택 및 학습

# num_leaves는 LightGBM에서 사용하는 파라미터 중 하나로, 
# 하나의 트리가 가질 수 있는 최대 리프(leaf) 노드의 개수를 지정
# num_leaves의 값이 크면 모델이 더 복잡해지고 더 많은 학습을 할 수 있게 되지만, 
# 과적합(overfitting)의 위험이 증가합니다. 
# 반대로, 이 값이 너무 작으면 모델이 너무 단순해져서 학습이 충분히 이루어지지 않을 수 있음


## CODE
params = {
    'boosting_type': 'gbdt', # Gradient Boosting Decision Tree
    'objective': 'multiclass', # 다중 클래스 분류
    'metric': 'multi_logloss', # 평가 지표
    'num_leaves': 31, # 한 트리가 가질 수 있는 최대 리프 수
    'learning_rate': 0.05, # 학습률
    'feature_fraction': 0.9, # 트리를 생성할 때마다 선택할 피처의 비율
    'bagging_fraction': 0.8, # 데이터를 샘플링하는 비율
    'bagging_freq': 5, # k회 반복마다 배깅을 수행
    'num_class': len(np.unique(y_train)), # 클래스의 총 수
    'verbose':0, # cotrol log information 
}

# 모델 학습
gbm = lgb.train(params,
                train_dataset,
                num_boost_round=200, # 부스팅 반복 횟수
                valid_sets=test_dataset, # 평가 데이터셋
                early_stopping_rounds=20) # 조기 종료 조건

# 3. 예측 및 평가
# 회귀 문제에서도 예측과 평가는 분류 문제와 유사한 절차를 따르며, 평가 지표만 회귀 분석에 적합한 것으로 변경합니다. 
# 예를 들어, RMSE(평균 제곱근 오차)나 MAE(평균 절대 오차) 등을 사용할 수 있습니다.

## CODE
# 예측
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
predicted_classes = np.argmax(y_pred, axis=1)

# 성능 평가 (예: 정확도 계산)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_classes)
print(f'Test Accuracy: {accuracy}')
