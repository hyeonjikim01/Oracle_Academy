## 유방암 데이터셋 불러오기
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)



## 기본 유방암 예측 모델(logistic regression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler로 데이터 전처리 (0~1 스케일)
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x) # 학습 데이터에 fit + transform
test_x_scaled = scaler.transform(test_x) # 테스트 데이터는 transform만

# 기본 Logistic Regression 모델 생성
logistic_regression = LogisticRegression(max_iter = 100)

# regression 학습
# logistic_regression.fit(train_x, train_y)
logistic_regression.fit(train_x_scaled, train_y)

# 테스트 데이터 예측
# pred = logistic_regression.predict(test_x)
pred = logistic_regression.predict(test_x_scaled)

# 정확도 계산
acc = accuracy_score(test_y, pred)
print("breast_cancer Accuracy: ", acc)



## L1 정규화를 적용한 유방암 예측 모델
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# L1 정규화를 적용한 로지스틱 회귀 모델 생성
selector = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.1, solver = 'liblinear', max_iter = 100))

# 모델 학습
# selector.fit(train_x, train_y)
selector.fit(train_x_scaled, train_y)

# 중요한 특성만 선택
# X_train_new = selector.transform(train_x)
# X_test_new = selector.transform(test_x)
X_train_new = selector.transform(train_x_scaled) # L1 정규화를 통해 살아남은 열만 남음
X_test_new = selector.transform(test_x_scaled)

# 새로운 특성으로 모델 학습 및 예측
model = LogisticRegression()
model.fit(X_train_new, train_y)
y_pred = model.predict(X_test_new)

# 정확도 계산
acc = accuracy_score(test_y, y_pred)
print("L1 Accuracy: ", acc)



## L2 정규화를 적용한 유방암 예측 모델
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# L2 정규화를 적용한 로지스틱 회귀 모델 생성
selector = SelectFromModel(LogisticRegression(penalty = 'l2', C = 0.1, solver = 'liblinear', max_iter = 100))

# 모델 학습
# selector.fit(train_x, train_y)
selector.fit(train_x_scaled, train_y)

# 중요한 특성만 선택
# X_train_new = selector.transform(train_x)
# X_test_new = selector.transform(test_x)
X_train_new = selector.transform(train_x_scaled) # L2 정규화를 통해 살아남은 열만 남음
X_test_new = selector.transform(test_x_scaled)

# 새로운 특성으로 모델 학습 및 예측
model = LogisticRegression()
model.fit(X_train_new, train_y)
y_pred = model.predict(X_test_new)

# 정확도 계산
acc = accuracy_score(test_y, y_pred)
print("L2 Accuracy: ", acc)