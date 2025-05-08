# ## K-Fold Cross-Validation을 활용한 유방암 진단 모델
#
# # 라이브러리 불러오기
# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
# import numpy as np
#
# # 데이터 불러오기
# cancer = load_breast_cancer()
# x = cancer.data
# y = cancer.target
#
# # K-Fold Cross Validation 설정 (k = 5)
# kf = KFold(n_splits=5)
#
# # 정확도를 저장할 리스트 초기화
# acc_list = []
#
# # K-Fold Cross Validation 수행
# for train_index, test_index in kf.split(x):
#     train_x, test_x = x[train_index], x[test_index]
#     train_y, test_y = y[train_index], y[test_index]
#
#     # 매 fold마다 모델 새로 생성 (기존 인스턴스를 재사용하지 않도록)
#     logistic = LogisticRegression(max_iter=10000) # max_iter를 늘려 수렴 문제 해결
#     logistic.fit(train_x, train_y)
#     pred = logistic.predict(test_x)
#     acc = accuracy_score(test_y, pred)
#     acc_list.append(acc)
#
# # 평균 정확도 계산
# mean_acc = np.mean(acc_list) # 최종적으로 5개의 폴드에 대한 평균 정확도를 계산
# print("평균정확도 = ", mean_acc)



## logistic regression -> SVM으로 모델을 변경

# 라이브러리 및 데이터 로드
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC

# 데이터 불러오기
cancer = load_breast_cancer()
x = cancer.data # 특성 데이터 (입력값, shape: [샘플 수, 특성 수])
y = cancer.target # 타깃 데이터 (레이블, 0 = 악성, 1 = 양성)

# K-Fold Cross Validation 설정 (k = 5)
kf = KFold(n_splits=5) # 데이터를 5개의 폴드로 나누어 5번의 학습/테스트 반복

# 모델 초기화
# logistic_regression = LogisticRegression(max_iter = 10000)
svc = SVC(kernel = 'rbf', max_iter=1000000)  # RBF 커널을 사용하는 SVM 모델 생성, 반복 횟수 최대 100만으로 설정

# 정확도를 저장할 리스트 초기화
acc_list = [] # 각 fold의 정확도를 저장하기 위한 리스트

# K-Fold Cross Validation 수행
for train_index, test_index in kf.split(x):
    train_x, test_x = x[train_index], x[test_index]
    train_y, test_y = y[train_index], y[test_index]

    svc.fit(train_x, train_y) # SVM 모델 학습 (훈련 데이터에 대해 학습)
    pred = svc.predict(test_x) # 테스트 데이터에 대해 예측 수행
    acc = accuracy_score(test_y, pred) # 예측 결과와 실제 값 비교하여 정확도 계산
    acc_list.append(acc) # 정확도를 리스트에 추가

# 평균 정확도 계산
mean_acc = np.mean(acc_list) # 5개 fold에서 계산된 정확도의 평균값
print('평균정확도 = ', mean_acc) # 평균 정확도 출력
