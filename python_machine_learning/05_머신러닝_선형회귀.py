## Linear Regression 사용법 실습
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# # 학습 데이터 생성
# x = np.array([[1], [2], [3], [4], [5]])
# y = np.array([4, 6, 8, 10, 12])
#
# # Linear Regression 모델 학습
# linear_regression = LinearRegression()
# linear_regression.fit(x, y)
# pred = linear_regression.predict([[6]]) # x가 6일 때 예측
#
# # 기울기, 절편, 예측값 출력
# print("기울기:", linear_regression.coef_)       # 이울기 : [2.]
# print("절편:", linear_regression.intercept_)     # 절편 : 2.0 -> 상수항
# print("6에 대한 예측:", pred)                    # 6에 대한 예측 [14.]
#
# # 그래프 시각화
# plt.plot(x, y, 'r')  # 실제 데이터 (빨간 선)
# plt.show()



## 혈당 예측 모델

from sklearn.datasets import load_diabetes

# 데이터 불러오기
diabetes = load_diabetes()
x = diabetes.data # 입력 데이터
y = diabetes.target # 타겟 데이터
print("feature names :", diabetes.feature_names)
# feature names : ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print("data size = ", x.shape)
print("target size = ", y.shape)

# 학습 80%, 테스트 20%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 데이터 학습
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2) # 랜던시드 고정하지 않아서 결과 계속 바뀔 수 있음
linear_regression = LinearRegression()
linear_regression.fit(train_x, train_y)

# 결과예측
from sklearn.metrics import mean_absolute_error

pred = linear_regression.predict(test_x)
mae = mean_absolute_error(test_y, pred)
print("MAE: ", mae)

# 산점도로 실제 값 vs 예측 값 시각화
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정 (예: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

# 산점도 시각화
plt.figure(figsize=(8, 6))
plt.scatter(test_y, pred, color='skyblue', edgecolor='k', alpha=0.7)
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='-', label='이상적인 예측')
plt.xlabel("실제 값")
plt.ylabel("예측 값")
plt.title("당뇨병 예측 결과 비교")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## 결정계수 확인하기
from sklearn.metrics import r2_score
r2 = r2_score(test_y, pred)
print("r2 : ", r2)