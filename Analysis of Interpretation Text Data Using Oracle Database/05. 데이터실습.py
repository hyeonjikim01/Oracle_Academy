## 데이터 불러오기

import pandas as pd
import re
import oracledb
import os

oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_23_8")

connection = oracledb.connect(
    user="EDU119",
    password="oracle_4U",
    dsn="138.2.63.245:1521/srvinv.sub03250142080.kdtvcn.oraclevcn.com"
)

sql = "SELECT * FROM 미생물배양검사"
df = pd.read_sql(sql, con=connection)

df.to_excel("미생물배양검사_결과.xlsx", index=False)
# print(os.getcwd())

#### 2. 자유 텍스트 판독문 처리 ####
## 비정형 데이터 전처리

df["검사결과"] = df["검사결과"].fillna("").str.replace(r"[\r\n]+", " ", regex = True).str.replace(r"\s{2,}", " ", regex = True)
    # \r\n은 줄바꿈된 부분을 " "로 변환
    # \s{2,} 빈 칸이 두 개 이상 반복될 경우 한 칸으로 변환


##항생제 목록 입력

ANTIBIO_LIST={'Amikacin':'AM','Amoxicillin/Clavulanic':'AMC','Ampicillin':'AM.1','ESBL':'ESBL',
    'Imipenem':'IPM','Tigecycline':'TIG','Vancomycin':'VA','Ampicillin/Sulbactam':'SAM',
    'Trimethoprim/Sulfamethoxazole':'SXT','Gentamicin':'GM','Ciprofloxacin':'CIP','Minocycline':'MI',
    'Meropenem':'MEM','Cefepime':'FEP','Ceftazidime':'CTX','Cefotaxime':'CTX.1','Piperacillin/Tazobactam':'TZP',
    'Piperacillin':'PIP','Ticarcillin/Clavulanic':'TIM','Colistin':'CL','Aztreonam':'ATM','Ertapenem':'ERT',
    'Cefoxitin':'FOX','Cefazolin':'CZ','Chloramphenicol':'C','Tetracycline':'TE','Benzylpenicillin':'P-G',
    'Clindamycin':'CC','Linezolid':'LNZ','Oxacillin':'OX.1','Nitrofurantoin':'NIT',
    'Quinupristin/Dalfopristin':'QDA','Fusidic':'FA','Habekacin':'HAB','Rifampicin':'RIF','Erythromycin':'E',
    'Tobramycin':'TOB','Netilmicin':'NET','Moxifloxacin':'MXF','Ceftriaxone':'CRO','Norfloxacin':'NOR'}


## 감염균 추출 함수

# # 동정결과 패턴 반복시
# i = 0
# while i < len(df): # while로 데이터프레임 한식 순회
#     row = df.iloc[i] # i번째 행
#     result = str(row["검사결과"]) # 문자열로 변환
#     if result.count("동정결과") >= 2: # 두 번 이상 나오면
#         parts = result.split("동정결과", maxsplit=2) # 동정 결과 기준으로 최대 두 번 분리(0, 1, 2) -> 이전/사이/이후
#         df.at[i, "검사결과"] = parts[0].strip() + "\n동정결과" + parts[1].strip() # 동정결과 첫 번째 기준으로 잘라 재구성
#         new_row = row.copy() # 기존 행 복사
#         new_row["검사결과"] = "동정결과" + parts[2].strip() # 두 번째 정보 포함된 부분
#         df = pd.concat([ # 새로운 행을 i 다음에 삽입
#             df.iloc[:i+1], # 현재 행까지 유지
#             pd.DataFrame([new_row]), # 새로운 행 삽입
#             df.iloc[i+1:] # 나머지 행 유지
#         ], ignore_index=True) # 인덱스 재정렬
#         i += 1 # 새 행 삽입해서 한 번만 증가
#     else:
#         i += 1 # 동정결과 1번 이하일 경우 다음행으로 이동

# 이전 버전
# def extract_bacteria(text):
#     if not isinstance(text, str): # 문자열이 아니면 None 반환
#         return None
#     match = re.search(r"동정결과[:\-]?\s*([A-Za-z\.\-]+(?:\s[A-Za-z\.\-]+)?)", text)
#     if match: # 감염균 이름 있는 경우 -> re.search가 성공이면
#         return match.group(1).strip() # 첫 번째 괄호에 매칭된 그룹을 .strip() 추출해서 앞뒤 공백 제거 후 반환
#     return None # 동정결과가 없으면 None 반환


# 수정 버전
i = 0
while i < len(df):
    row = df.iloc[i]
    result = str(row["검사결과"])
    if result.count("동정결과") >= 2:
        parts = result.split("동정결과", maxsplit=2)
        df.at[i, "검사결과"] = parts[0].strip() + "\n동정결과" + parts[1].strip()
        new_row = row.copy()
        new_row["검사결과"] = "동정결과" + parts[2].strip()
        df = pd.concat([
            df.iloc[:i+1],
            pd.DataFrame([new_row]),
            df.iloc[i+1:]
        ], ignore_index=True)
        i += 1
    else:
        i += 1


def extract_bacteria(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"동정결과[:\-]?\s*([A-Za-z\.\-]+(?:\s[A-Za-z\.\-]+)?)", text)
    if match:
        return match.group(1).strip()
    if "stenotrophomonas maltophilia" in text.lower():
        return "Stenotrophomonas maltophilia"
    if "klebsiella pneumoniae" in text.lower():
        return "Klebsiella pneumoniae"
    return None


def extract_antibiotics(text, pattern):
    found = []
    for abx in ANTIBIO_LIST:
        regex = (
            rf"{abx}\s*:?\s*(?:<=|>=|<|>|=)?\s*\d*\.?\d*\s*\(\s*{pattern}\s*\)"
            rf"|{abx}\s*:?\s*{pattern}\s*"
            rf"|{abx}\s+Neg\s+\(\s*{pattern}\s*\)"
        )
        matches = re.findall(regex, text, re.IGNORECASE)
        if matches:
            found.append(abx)
    return list(set(found)) if found else None



# print(df.columns)
# print(df.head())

# df.to_excel("df4.xlsx", index = False)


## 항생제 정보 추출 함수

# 이전 버전
# def extract_antibiotics(text, pattern): # 특정 패턴 만족하는 항생제 이름 추출
#     found = [] # 항생제 이름 저장 리스트
#     for abx in ANTIBIO_LIST: # 딕셔너리 중 키값만 출력, abx는 항생제 이름
#         regex = ( # 세 가지 pattern으로 작성된 항생제 감수성 결과 포착
#             rf"{abx}\s*:?\s*(?:<=|>=|<|>|=)?\s*\d*\.?\d*\s*\(\s*{pattern}\s*\)"
#             # \s*:?\s* 양옆 공백 허용, (?:)는 비캡처그룹, \d*\.?\d* 소숫점 포함한 숫자
#             rf"|{abx}\s*:?\s*\(\s*{pattern}\s*\)"
#             # \s*:?\s 공백 콜론 공백, \(\s*{pattern}\s*\) 괄호 안에 패턴
#             rf"|{abx}\s+Neg\s+\(\s*{pattern}\s*\)"
#             # \s+Neg\s+ Neg후 공백, \(\s*{pattern}\s*\) 괄호 안에 패턴
#         )
#         matches = re.findall(regex, text, re.IGNORECASE) # re.findall로 정규표현식에 모두 매칭되는 부분을 리스트로 반환
#         if matches: # 항생제가 있다면
#             found.append(abx) # found 리스트에 추가
#     return list(set(found)) if found else None # 리스트 중복 제거 후 리스트로 변환하여 반환, 없으면 None

# 결과 테이블 생성 함수
df["감염균"] = df["검사결과"].apply(extract_bacteria) # .apply는 판다스에서 각 행이나 열에 함수 적용할 때 쓰는 것. 즉, extract~에 함수 적용
df["항생제_S"] = df["검사결과"].apply(lambda x: extract_antibiotics(x, "S"))
df["항생제_R"] = df["검사결과"].apply(lambda x: extract_antibiotics(x, "R"))
df["항생제_I"] = df["검사결과"].apply(lambda x: extract_antibiotics(x, "I"))

df.to_excel("미생물배양검사_정리.xlsx", index = False)


#### 3. 워드 임베딩 전처리 ####
## 판정 결과를 문장으로 변환
df2 = df[df["감염균"].notna()] # 감염군이 있는 행만 필터링해서 df2에 저장

def combine_antibiotics(row):
    text = []
    for abx in row["항생제_S"] or []:
        text.append(f"{abx}_S")
    for abx in row["항생제_R"] or []:
        text.append(f"{abx}_R")
    for abx in row["항생제_I"] or []:
        text.append(f"{abx}_I")
    return " ".join(text) # 공백으로 연결된 문장

df2["임베딩_문장"] = df2.apply(combine_antibiotics,axis = 1) # axis = 1 열 기준


#### 4. 워드 임베딩 ####
## 모형 적합
sentences = df2["임베딩_문장"].apply(lambda x: x.split()).tolist()

from gensim.models import Word2Vec

#  Word2Vec 학습
model = Word2Vec(
    sentences,
    vector_size = 100, # 벡터 차원 수
    window = 5, # 컨텍스트 윈도우 크기 -> 주변 단어를 몇 개까지 고려
    min_count = 1, # 최소 등장 횟수 -> 해당 값 보다 적게 등장한 단어는 무시
    workers = 4, # 병렬 처리 쓰레드 수
    sg = 1 # Skip-gram 모델 (0이면 CBOW) -> 학습 방식 선택
)

## 유사도 점수
print(model.wv.most_similar(["Ciprofloxacin_R"]))
    # model.wv 모델 안의 벡터에 접근
    # .most_similar() 특정 단어와 가장 유사한 단어 10개 반환
    # ["Ciprofloxacin_R"] 기준이 되는 단어 -> 이것과 유사한 것 선택


# ## PAC를 이용한 시각화
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
#
# words = list(model.wv.key_to_index.keys())[:40] # 너무 많으면 겹치기 때문에 40개만 선택
# vectors = [model.wv[word] for word in words] # 각 단어를 Word2Vec에서 꺼내서 벡터로 저장
#
# pca = PCA(n_components=2) # 두개 주성분으로 축소(PC1, PC2)
# pca_result = pca.fit_transform(vectors)
#
# plt.figure(figsize=(12, 8))
# plt.scatter(pca_result[:, 0], pca_result[:, 1])
#
# for i, word in enumerate(words):
#     plt.text(pca_result[i, 0] + 0.01, pca_result[i, 1], word, fontsize = 9)
#     # 각 점에 라벨 붙이기, +0.01은 점과 텍스트가 겹치지 않게 살짝 밀기
#
# plt.title("ABX & Patterns")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.show()


# ## TSNE를 이용한 시각화
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
#
# words = list(model.wv.index_to_key) # 학습된 단어 목록 리스트로 가져오기
# word_vectors = model.wv[words] # 각 단어의 벡터 가져오기
#
# tsne = TSNE(n_components = 2, random_state = 42, perplexity=15, n_iter = 1000)
# word_vec_2d = tsne.fit_transform(word_vectors)
#
# plt.figure(figsize=(15, 10))
# for i, word in enumerate(words): # 인덱스, 단어 가져오기
#     x, y = word_vec_2d[i] # 2차원 벡터 형태 [x, y]
#     plt.scatter(x, y)
#     plt.text(x + 0.1, y + 0.1, word, fontsize = 9)
# plt.title("Word2Vec Embedding (ABX + Pattern)")
# plt.xlabel("TSNE-1")
# plt.ylabel("TSNE-2")
# plt.grid(True)
# plt.show()


## K-means를 이용한 군집화
# # 벡터들을 K-means로 군집화하고 PCA 이용해 2차원으로 축소한 뒤 시각화
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
# words = list(model.wv.key_to_index)
# vectors = [model.wv[word] for word in words] # 각 단어의 벡터를 리스트에 저장
#
# kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
# labels = kmeans.fit_predict(vectors)
#
# pca = PCA(n_components=2) # 두개 주성분으로 축소(PC1, PC2)
# pca_result = pca.fit_transform(vectors)
#
# plt.figure(figsize=(14, 9))
#
# for i, word in enumerate(words):
#     x, y = pca_result[i]
#     plt.scatter(x, y, c = f"C{labels[i]}",
#                 label = f"Cluster {labels[i]}" if i == 0 else "", alpha = 0.8)
#     plt.text(x + 0.01, y + 0.01, word, fontsize = 9)
# plt.title("KMeans (ABX+patterns)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.show()


# 3차원 그래프
# from sklearn.cluster import KMeans
# import plotly.express as px
# import pandas as pd
# from sklearn.decomposition import PCA
#
# # 벡터 준비
# words = list(model.wv.key_to_index.keys())#[:100]
# vectors = [model.wv[word] for word in words]
#
# # 차원 축소
# pca = PCA(n_components=3) # 3차원
# pca_result = pca.fit_transform(vectors)
#
# # 클러스터링
# kmeans = KMeans(n_clusters=5, random_state=42)
# labels = kmeans.fit_predict(pca_result)
#
# # 데이터프레임 생성
# df_plot = pd.DataFrame(pca_result, columns=["x", "y", "z"])
# df_plot["word"] = words
# df_plot["cluster"] = labels.astype(str)  # 문자열로 변환하면 색 구분에 더 유리
#
# # 시각화
# fig = px.scatter_3d(
#     df_plot, x="x", y="y", z="z",
#     text="word", color="cluster"
# )
# fig.update_traces(marker=dict(size=5))
# fig.write_html("clustered_plot.html")
# os.startfile("clustered_plot.html")


#### 5. 모델 개발 및 학습 ####
import numpy as np

def sentence_vector(sentence, model):
    words = sentence.split() # 문장을 단어 리스트로 분리
    vectors = [model.wv[word] for word in words if word in model.wv] # Word2Vec모델에 존재하는 단어만 필터링
    if vectors:
        return np.mean(vectors, axis = 0) # 단어 벡터들의 평균을 문장 벡터로 변환
    else:
        return np.zeros(model.vector_size) # 단어 하나도 없을 경우, 0벡터 반환
top_n = 10 # 원하는 개수로 설정

top_classes = df2["감염균"].value_counts().nlargest(top_n).index.tolist()
df3 = df2[df2["감염균"].isin(top_classes)] # 상위 10개만 남김

X = np.array([sentence_vector(s, model) for s in df3["임베딩_문장"]])
y = df3["감염균"].values

## Random Forest
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_rf.fit(X_train, y_train)
#
# y_pred = clf_rf.predict(X_test)
# print("Random Forest")
# print(classification_report(y_test, y_pred))

# # 시각화
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# y_pred_rf = clf_rf.predict(X_test) # 예측 수행
#
# labels_ordered = sorted(list(set(y_test))) # 클래스 이름 정렬
#
# cm = confusion_matrix(y_test, y_pred_rf, labels = labels_ordered) # 혼동행렬 계산
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels_ordered, yticklabels=labels_ordered)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix: Random Forest")
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
# plt.tight_layout()
# plt.show()


# ## Logistic Regression
# from sklearn.linear_model import LogisticRegression
#
# clf_lr = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
# clf_lr.fit(X_train, y_train)
# y_pred_lr = clf_lr.predict(X_test)
#
# print("Logistic Regression")
# print(classification_report(y_test, y_pred_lr))
#
# # 시각화
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# y_pred_lr = clf_lr.predict(X_test) # 예측 수행
#
# labels_ordered = sorted(list(set(y_test))) # 클래스 이름 정렬
#
# cm = confusion_matrix(y_test, y_pred_lr, labels = labels_ordered) # 혼동행렬 계산
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels_ordered, yticklabels=labels_ordered)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix: Logistic Regression")
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
# plt.tight_layout()
# plt.show()


## MLP
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
clf_mlp.fit(X_train, y_train)

y_pred_mlp = clf_mlp.predict(X_test)
print("MLP (StandardScaler + max_tier=1000)")
print(classification_report(y_test, y_pred_mlp, zero_division=0))

labels_ordered = sorted(list(set(y_test)))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_mlp, labels=labels_ordered)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", xticklabels=labels_ordered, yticklabels=labels_ordered)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: MLP(StandardScaler)")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()