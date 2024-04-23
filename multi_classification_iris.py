# 0.
# pip install lightgbm==3.3.5

# 1. 데이터 준비
# 데이터 준비 단계는 분류 문제와 동일하게, 특성(features)과 타겟 변수(target)를 준비하고, LightGBM의 데이터셋 형식으로 변환합니다.

## CODE
from sklearn import datasets
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np

# Iris 데이터셋 로드
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)


# 2. 모델 학습

# metric='multi_logloss' : Multiclass logarithmic loss
## Multiclass logarithmic loss은 다중 클래스 분류 문제에 적용된 cross entropy 손실의 한 형태
# feature_fraction : 트리를 훈련시킬 때마다 무작위로 선택되는 피처의 비율
## feature_fraction=0.9 -> 각 반복에서 사용되는 피처의 90%만을 무작위로 선택 및 학습
# bagging_fraction  : 데이터의 일부분만을 사용하여 트리를 훈련시키는 데 사용
## bagging_fraction=0.8 -> 전체 데이터 세트의 80%를 무작위로 선택하여 트리 학습에 사용
# bagging_freq는 bagging을 수행할 빈도
## bagging_freq는=5 -> 5번의 부스팅 반복마다 새로운 데이터의 부분 집합이 선택 및 학습



## CODE
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'num_class': len(np.unique(y_train)), # 클래스의 총 수
    'verbose': 0
}

gbm = lgb.train(params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, test_data],
                early_stopping_rounds=10)



# 3. 예측 및 평가

## CODE
# 예측
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
predicted_classes = np.argmax(y_pred, axis=1)

# 성능 평가
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_classes)
print(f'Test Accuracy: {accuracy}')
