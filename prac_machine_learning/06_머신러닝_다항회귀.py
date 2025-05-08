# ## Polynomial Linear Regression 사용법
#
# # 파이썬 버전이 3.7 이상인지 확인
# import sys
# assert sys.version_info >= (3, 7)
#
# # numpy, LinearRegression과 PolynomialFeatures 불러오기
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
#
# # 학습 데이터 생성
# x = np.array([[1], [2], [3], [4], [5]])
# y = np.array([2, 4, 6, 8, 10])
#
# # 데이터를 다항식으로 변환, include_bias는 절편을 추가하는 파라미터
# poly = PolynomialFeatures(degree=2, include_bias=False)
#     # 각 데이터에 대해 다항식 조합을 추가한 새로운 특성 행렬을 생성
#     # degree = 2일 때, [x1, x2, x1^2, x1*x2, x2^2]
# poly.fit(x) # feature의 수를 계산
# x_transformed = poly.transform(x) # x를 2차식으로 변환하는 함수
#
# # Linear Regression 모델 학습
# linear_regression = LinearRegression()
# linear_regression.fit(x_transformed, y)
# x_test = [[6]] # x에 6을 넣으면
# x_test_transformed = poly.transform(x_test)
# pred = linear_regression.predict(x_test_transformed)
#
# print("기울기: ", linear_regression.coef_)
# print("절편: ", linear_regression.intercept_)
# print("6에 대한 예측: ", pred)



## 당뇨병 환자의 혈당 예측 Linear Regression 사용법

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures

diabetes = load_diabetes()
x = diabetes.data # 입력 데이터
y = diabetes.target # 타켓 데이터

print("feature names :", diabetes.feature_names)
print("data :", x.shape)
print("target :", y.shape)

# 데이터를 다항식으로 변환
poly = PolynomialFeatures(degree = 2, include_bias=False) # 2차함수로 확장, 절편 항은 생성X
poly.fit(x) # feature 수를 계산
x_transformed = poly.transform(x) # x를 2차식으로 변환하는 함수

# Linear Regression 모델 학습
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_transformed, y)
print('x_test[0] = ', x_transformed[[0]])

x_test = x_transformed[[0]] # 첫번째 샘플을 2차원 배열로 추출
pred = linear_regression.predict(x_test)
print('Polynomial Features 당뇨병 환자 혈당 예측 = ', pred)



## 당뇨병 데이터셋 dataset split

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# 데이터 불러오기
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

print("feature names :", diabetes.feature_names)  # 특성 이름
print("data :", x.shape)  # (442, 10)
print("target :", y.shape)  # (442,)

# 다항식 특성 변환
poly = PolynomialFeatures(degree=2, include_bias=False)  # 절편은 제외
poly.fit(x)  # 특성 조합 학습
x_transformed = poly.transform(x)  # 원본 데이터를 다항식으로 확장

# 학습/테스트 데이터 분리 (80:20)
train_x, test_x, train_y, test_y = train_test_split(x_transformed, y, test_size=0.2)



## 혈당 예측 모델 - Polynomial Linear Regression

# Linear Regression Fit
linear_regression = LinearRegression()
linear_regression.fit(train_x, train_y)

# 결과 예측
from sklearn.metrics import mean_absolute_error

# 예측 수행
pred = linear_regression.predict(test_x)

# MAE 측정
mae = mean_absolute_error(test_y, pred)
print("MAE: ", mae)


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정 (예: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

# 산점도: 실제 값 vs 예측 값
plt.figure(figsize=(8, 6))
plt.scatter(test_y, pred, color='dodgerblue', alpha=0.7, label='예측 값')

# 이상적인 예측선 (y=x)
min_val = min(test_y.min(), pred.min())
max_val = max(test_y.max(), pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=2, label='이상적인 예측')


# 그래프 레이블 및 제목 설정
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.title('실제 값과 예측 값 비교')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
