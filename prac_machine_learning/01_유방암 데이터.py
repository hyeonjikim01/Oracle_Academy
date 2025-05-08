from sklearn import datasets
import pandas as pd

# load data
cancer = datasets.load_breast_cancer()

# 데이터셋 구조 확인
print(cancer.keys())

# 특성 데이터와 타겟 데이터 확인
print("특성 이름:\n", cancer.feature_names)
print("타겟 이름:\n", cancer.target_names)

# 특성 데이터 (앞의 5개)와 타겟 데이터 확인
print("특성 데이터(Feature):\n", cancer.data[:5])
print("특성 데이터(Target):\n", cancer.data[:5])

# Pandas DataFrame으로 변환
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df['target'] = cancer.target

# 데이터 프레임 상위 5개 행 확인
print(df.head())

a = 1



### 회귀모델과 성능평가 ####

# scikit-learn 라이브러리에서 mean_absolute_error 불러오기
from sklearn.metrics import mean_absolute_error

y = [2, 3, 4, 1, 5]
y_pred = [1.5, 2.5, 5, 2, 4.5]

mae = mean_absolute_error(y, y_pred)
print("mae : ", mae) # mae : 0.7

# scikit-learn 라이브러리에서 mean_squared_error 불러오기
from sklearn.metrics import mean_squared_error

y = [2, 3, 4, 1, 5]
y_pred = [1.5, 2.5, 5, 2, 4.5]

mse = mean_squared_error(y, y_pred)
print("mse : ", mse) # mse : 0.55

# R^2 불러오기
from sklearn.metrics import r2_score

y = [2, 3, 4, 1, 5]
y_pred = [1.5, 2.5, 5, 2, 4.5]





### 분류모델과 성능평가 ####

## Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y = [1, 0, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0]

cm = confusion_matrix(y, y_pred)

print("confusion matrix : ", cm)
sns.heatmap(cm, annot = True, cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


## Accuracy(정확도)
# scikit - learn 라이브러리에서 accuracy_score 불러오기
from sklearn.metrics import accuracy_score

y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]

acc = accuracy_score(y, y_pred)
print("accuracy : ", acc) # accuracy : 0.6


## Precision(정확도)
# scikit-learn 라이브러리에서 preciion_score 불러오기
from sklearn.metrics import precision_score

y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]

precision = precision_score(y, y_pred)

print("precision : ", precision) # precision : 0.666


## Recall
# scikit-learn 라이브러리에서 recall 불러오기
from sklearn.metrics import recall_score

y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]

recall = recall_score(y, y_pred)

print("recall : ", recall) # recall : 0.666


## F1 socre
from sklearn.metrics import f1_score

y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]

f1 = f1_score(y, y_pred)

print("f1_score : ", f1) # f1 : 0.666


## ROC curve 그리기
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y = [0, 0, 0, 1, 1]
y_pred = [0.1, 0.4, 0.7, 0.6, 0.8]

fpr, tpr, threshold = roc_curve(y, y_pred)

print("fpr : ", fpr) # fpr : [0. 0. 0.33 0.33 1.]
print("tpr : ", tpr) # tpr : [0. 0.5 0.5 1. 1.]
print("threshold : ", threshold) # thresholds : [1.8 0.8 0.7 0.6 0.1]

fig = plt.figure(figsize = (8, 8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(fpr, tpr, label = 'AdaBoost')
ax.plot([0, 1], [0, 1], color = 'red', label = 'Random Model')
ax.legend()
plt.show()


## AUROC
# scikit-learn 라이브러리에서 roc_auc_score 불러오기
from sklearn.metrics import roc_auc_score

y = [0, 0, 0, 1, 1]
y_pred = [0.1, 0.4, 0.7, 0.6, 0.8]

auroc = roc_auc_score(y, y_pred)

print("AUROC : ", auroc) # AUROC : 0.833