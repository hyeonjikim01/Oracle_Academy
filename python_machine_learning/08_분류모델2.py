# ## SVM 기본 사용법
#
# # numpy와 SVC 불러오기
# import numpy as np
# from sklearn.svm import SVC
#
# # 학습 데이터 생성
# X = np.array([[1], [2], [3], [4], [5]])
# y = np.array([0, 0, 0, 1, 1])
#
# # Linear Regression 모델 학습
# svc = SVC()
# svc.fit(X,y)
#
# pred = svc.predict([[6]])
# print("6에 대한 예측 : ", pred) # 6에 대한 예측 : [1]
#
#
#
# ## 유방암 분류 모델
#
# # 데이터셋 불러오기
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
#
# cancer = load_breast_cancer()
# x = cancer.data
# y = cancer.target
#
# print("feature names : ", cancer.feature_names)
# print("data : ", x.shape)       # data : (569, 30)
# print("target : ", y.shape)     # target : (569, )
#
# # 데이터 분할 train 80%, test 20%
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # 데이터 스케일 (0~1 스케일)
# from sklearn.preprocessing import MinMaxScaler
#
# scaler = MinMaxScaler()
# train_x_scaled = scaler.fit_transform(train_x)  # 학습 데이터는 fit + transform
# test_x_scaled = scaler.transform(test_x)        # 테스트 데이터는 transform만
#
# # 모델 구성 및 학습
# svc_scale = SVC(max_iter = 1000)
# svc_scale.fit(train_x_scaled, train_y)
#
# # 결과 예측
# from sklearn.metrics import accuracy_score
# pred = svc_scale.predict(test_x_scaled) # test set 예측
# acc = accuracy_score(test_y, pred) # accuracy 수치 뽑기
# print("Scaled Test Set Accuracy: ", acc) # Accuracy 수치 확인
#
# #################################################################
# # Scaler를 적용하지 않을 경우
# svc = SVC(max_iter = 1000)  # 모델 선언
# svc.fit(train_x, train_y)   # 모델 학습
#
# # accuracy score 뽑기
# pred = svc.predict(test_x)
# acc = accuracy_score(test_y, pred)
# print("Scaled Test Set Accuracy: ", acc)



## Predicting Heart Failure

import numpy as np
from sklearn.svm import SVC
import pandas as pd

# 데이터 불러오기
df = pd.read_csv(r'../Heart Failure Clinical Records.csv')
df

a = 1