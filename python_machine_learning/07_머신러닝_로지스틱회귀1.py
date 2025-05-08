#### 시그모이드 함수 시각화 ####
# import numpy as np

# ## 가중치 w와 편향 b이 출력값에 어떤 영향을 미치는지 확인
#
# ## w값의 변화
#
# # 함수 정의
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(0.5 * x)
# y2 = sigmoid(x)
# y3 = sigmoid(2 * x)
#
# # 시각화
# import matplotlib.pyplot as plt
# plt.plot(x, y1, 'r', linestyle = '--') # w = 0.5
# plt.plot(x, y2, 'g') # w = 1
# plt.plot(x, y3, 'b', linestyle = '--')  # w = 2
# plt.plot([0,0], [1.0, 0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()


# ##  b값의 변화
#
# # 함수 정의
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(x + 0.5)
# y2 = sigmoid(x + 1)
# y3 = sigmoid(x + 1.5)
#
# # 시각화
# import matplotlib.pyplot as plt
# plt.plot(x, y1, 'r', linestyle = '--') # x + 0.5
# plt.plot(x, y2, 'g') # w = 1
# plt.plot(x, y3, 'b', linestyle = '--')  # x + 1.5
# plt.plot([0,0], [1.0, 0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()



## 10 이상은 1, 10 미만인 경우 0을 부여한 레이블 데이터

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 숫자 10부터 1

# 모델링
model = Sequential() # 선형모델 만들기
model.add(Dense(1, input_dim = 1, activation = 'sigmoid'))

sgd = optimizers.SGD(learning_rate = 0.01) # 러닝레이트 0.01
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

model.fit(x, y, epochs = 200) # 학습 200번

# 시각화
plt.plot(x, model.predict(x), 'b', x, y, 'k.')
plt.show()

print(model.predict([1, 2, 3, 4, 5]))
print(model.predict([11, 21, 31, 41, 500]))