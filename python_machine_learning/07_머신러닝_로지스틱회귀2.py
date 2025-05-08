## Logistic Regression 사용법

# numpy, LogisticRegression 불러오기
import numpy as np
from sklearn.linear_model import LogisticRegression

# 학습 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

# Logistic Regression 모델 학습
logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)
pred = logistic_regression.predict([[6]])
print("6에 대한 예측 : ", pred)  # 6에 대한 예측 : [1]



## Breast Cancer 예측 모델

# 데이터셋 불러오기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

print("feature names : ", cancer.feature_names)
print("data : ", x.shape)       # data : (569, 30)
print("target : ", y.shape)     # target : (569, )

# 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# 데이터 스케일 (0~1 스케일)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)  # 학습 데이터는 fit + transform
test_x_scaled = scaler.transform(test_x)        # 테스트 데이터는 transform만

# 모델 구성 및 학습
logistic_regression = LogisticRegression(solver = 'lbfgs', max_iter = 100)
logistic_regression.fit(train_x_scaled, train_y)

# 결과 예측(prediction) 및 정확도 평가
from sklearn.metrics import accuracy_score

pred = logistic_regression.predict(test_x_scaled)
acc = accuracy_score(test_y, pred)
print("Breast Cancer Accuracy = ", acc) # Accuracy : 0.96

