# #### 암 진단 텍스트 기반 키워드 추출 실습 ####
#
# import pandas as pd
# import re
# import oracledb
#
# oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_23_8")
# connection = oracledb.connect(
#     user="EDU119",
#     password="oracle_4U",
#     dsn="138.2.63.245:1521/srvinv.sub03250142080.kdtvcn.oraclevcn.com"
# )
#
#
# ## 데이터 불러오기
# sql = "SELECT * FROM KDT_CANCER"
# df = pd.read_sql(sql, con=connection)
#
# # 진단명 열 추출 (마지막 열 기준)
# diagnosis_col = df.columns[-1]
# diagnoses = df[diagnosis_col].astype(str).tolist()  # 진단명 열을 문자열 리스트로 추출
#
#
# ## 분류 기준 딕셔너리 (대분류: 키워드 목록)
# category_keywords = {
#     "담도암": ["cholangio", "duct", "vater"],
#     "방광암": ["bladder", "gallbladder", "gb"],
#     "유방암": ["breast", "유방", "pagets"],
#     "중추신경계암": ["nerve", "brain", "gliomatosis", "meningioma", "neurogenic", "pituitary", "spinal"],
#     "대장암": ["colon", "colonc", "cc", "대장", "sigmoid", "anal", "duodenal", "rectal", "splenic", "transverse"],
#     "식도암": ["esophageal", "esophagus", "식도"],
#     "부인과암": ["cervical", "cervix", "자궁", "ovarian", "endometrial", "endometrioid", "uterine", "uterus", "vaginal"],
#     "두경부암": ["oropharyngeal", "glottis", "larynx", "mouth", "oral", "squamous",
#                "nasal cavity", "tongue", "tonsilar", "sublingual", "glottic"],
#     "호지킨림프종": ["hodgkin"],
#     "백혈병": ["myeloid", "leukemia"],
#     "골수종": ["myeloma"],
#     "간암": ["hepatocellular", "hepato", "hcc"],
#     "폐암": ["lung", "폐암", "bronchial"],
#     "흑색종": ["melanoma", "흑색종", "basal", "bowen"],
#     "췌장암": ["pancrea"],
#     "전립선암": ["prostat", "전립선"],
#     "신장암": ["renal", "kidney", "ureter"],
#     "위암": ["gastric", "gc", "g cancer", "gast", "stomach", "위암", "liver"],
#     "갑상선암": ["thyr", "갑상선", "papillary"],
#     "비호지킨림프종": ["lymphoma", "mantle", "fungoides"],
#     "기타": []
# }
#
#
# ## 분류 함수 정의
# def classify_diagnosis(text):
#     text_lower = text.lower() # 소문자로 바꾸기
#     for category, keywords in category_keywords.items(): # category는 키, keywords는 값
#         for keyword in keywords:
#             if re.search(keyword, text_lower): # 정규표현식 사용해서 keyword가 text_lower 포함하는지 검사
#                 return category # 키워드가 하나라도 발견되면 즉시 카테고리 이름 반환
#     return "기타"
#
# df_yes = df[df["악성종양"] == "Yes"].copy() # 암 진단이 yes인 케이스만 추출
# df_yes["진단명 분류"] = df_yes[diagnosis_col].astype(str).apply(classify_diagnosis) # 진단명 분류 적용
#
# category_counts = df_yes["진단명 분류"].value_counts().reset_index() # 분류 결과 요약
# category_counts.columns = ["암 대분류", "건수"]
# print("\n 암대분류별 건수 요약 : \n", category_counts)
#
# ## 기타 항목 확인 함수
# def get_other_items(df, diagnosis_col, classified_col = "진단명 분류"):
#     df_etc = df[df[classified_col] == "기타"].copy()
#     etc_counts = df_etc[diagnosis_col].value_counts().reset_index()
#     etc_counts.columns = ["진단명", "건수"]
#     return etc_counts
#
# other_items = get_other_items(df_yes, diagnosis_col)
# print("\n 기타 분류 항목 : \n", other_items)


#### 실습과제_TF-IDF with Logistic Regression ####
import pandas as pd
import oracledb
import re
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 벡터화
from sklearn.linear_model import LogisticRegression           # 로지스틱 회귀 모델
from sklearn.pipeline import Pipeline                         # 파이프라인 연결
from sklearn.model_selection import train_test_split          # 데이터 분할 (train/test)
from sklearn.metrics import classification_report             # 평가 리포트


oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_23_8")
connection = oracledb.connect(
    user="EDU119",
    password="oracle_4U",
    dsn="138.2.63.245:1521/srvinv.sub03250142080.kdtvcn.oraclevcn.com"
)


## 데이터 불러오기
sql = """ SELECT * FROM KDT_CANCER WHERE "악성종양" = 'Yes' """
df_yes = pd.read_sql(sql, con=connection)

## 암 진단 대분류 키워드
category_keywords = {
    "Biliary": ["gallbladder", "bile duct", "cholangiocarcinoma", "klatskin"],
    "Bladder": ["bladder"],
    "Breast": ["breast", "mammary", "nipple"],
    "CNS": ["brain", "glioblastoma", "astrocytoma", "ependymoma"],
    "Colorectal": ["colon", "rectal", "sigmoid", "colorectal"],
    "Esophagus": ["esophagus"],
    "Gynecologic": ["cervix", "uterus", "ovary", "endometrial", "vaginal", "fallopian"],
    "Head & Neck": ["oral", "tongue", "pharynx", "larynx", "nasopharynx", "mandible", "neck"],
    "Leukemia": ["leukemia"],
    "Liver": ["liver", "hepatocellular", "hcc"],
    "Lung": ["lung", "pulmonary"],
    "Melanoma": ["melanoma"],
    "Myeloma": ["myeloma", "plasma cell"],
    "Pancreatic": ["pancreas", "pancreatic"],
    "Prostate": ["prostate"],
    "Renal": ["renal", "kidney", "wilms"],
    "Stomach": ["stomach", "gastric", "pyloric", "antrum"],
    "Thyroid": ["thyroid"],
    "Others": []
}


# 진단명을 대분류로 분류하는 함수
def classify_diagnosis(text):
    text_lower = text.lower()  # 입력된 텍스트를 소문자로 변환
    for category, keywords in category_keywords.items():  # 대분류별 키워드 순회
        for keyword in keywords:
            # 키워드가 텍스트에 포함되면 해당 카테고리 반환
            if re.search(keyword, text_lower, re.IGNORECASE):
                return category
    return "Others"  # 어떤 키워드에도 해당하지 않으면 'Others' 반환

df_yes["암_대분류"] = df_yes["암_진단명"].apply(classify_diagnosis)

df_train = df_yes[df_yes["암_대분류"] != "Others"] # Others 제외한 데이터만 사용해서 학습

## 스플릿 세팅 후 학습
# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    df_train["암_진단명"],
    df_train["암_대분류"],
    test_size=0.2,
    random_state=42
)

# TF-IDF + Logistic Regression 파이프라인 정의
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# 학습 수행
pipe.fit(X_train, y_train)


## 예측 및 성능평가
# 예측 수행
y_pred = pipe.predict(X_test)

# 분류 성능 평가
print(classification_report(y_test, y_pred))

# 결과를 데이터프레임으로 정리
results_df = pd.DataFrame({
    "진단명": X_test,
    "실제 분류": y_test.values,
    "예측 분류": y_pred
})

# 결과 출력
print(results_df)

# 판다스 출력 옵션 설정 (내용이 길어도 전부 보이도록)
pd.set_option('display.max_columns', None)   # 열 제한 없음
pd.set_option('display.max_rows', None)      # 행 제한 없음
pd.set_option('display.max_colwidth', None)  # 셀 내용 길이 제한 없음
pd.set_option('display.width', 1000)         # 줄바꿈 없이 넓게 표시

# 오분류된 샘플 확인
misclassified = results_df[results_df['실제 분류'] != results_df['예측 분류']]

# 결과 출력
print(misclassified)
