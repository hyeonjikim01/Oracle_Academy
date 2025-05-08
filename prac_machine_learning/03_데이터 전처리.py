# ## 유방암 데이터셋 Split
# # scikit-learn 라이브러리에서 train_test_split 불러오기
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# x = cancer.data
# y = cancer.target
# x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
#
# # x, y original dataset을 train셋과 valid셋으로 split 함
# print('split 전 유방암 x 데이터셋 크기', x.shape)
# print('split 전 유방암 y 데이터셋 크기', y.shape)
# print('split 후 x_train 데이터셋 크기', x_train.shape)
# print('split 후 x_valid 데이터셋 크기', x_valid.shape)
# print('split 후 y_train = ', y_train.shape)
# print('split 후 y_valid = ', y_valid.shape)
# print('-----------------------------------------')



# ## Z-score 정규화 실습
#
# # 데이터 준비
# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame({'x1' : np.arange(11), 'x2' : np.arange(11)**2}) # x1은 0~9로 채우고, x2는 0~9의 제곱수로 채움
#
# # z-score 정규화
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# df_std = scaler.fit_transform(df)
#
# z_df = pd.DataFrame(df_std, columns = ['x1_std', 'x2_std'])
# print(z_df)



## Min-Max 스케일링(Normalization)

import pandas as pd
import numpy as np
df = pd.DataFrame({'x1' : np.arange(11), 'x2' : np.arange(11)**2})

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = scaler.fit_transform(df)
pd.DataFrame(df_minmax, columns = ['x1_minmax', 'x2_minmax'])
print(df_minmax)
