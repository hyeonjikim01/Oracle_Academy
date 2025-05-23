#### 정규표현식 실습 ####

text = """
     [🔥충격] 국・비・부・트・캠・프・전・격・개・강!! 
     ▶ OR@CLE ACADEMY ※ 바이오헬♨스 특화 트♥랙
     ▶ 파☆이☆썬, S§Q§L, R, P@CS, 쭈피터 전.부.탑.재
     ▶ 합격시 교육비$$ 100% 즉/시무/료 $$ 매달 31만원 훈련장려금까지 증※정 
     ★ 의료 데이터부터 A/I까지 A to Z 전공자급 실무 커리큘럼 수강◎기회
     ★ 수강신청 폭주 중!! 선착순 ▼ ▼ ▼
     이외 궁금하신 내용은 ☏전화 02-6235-5168 또는 011-1234-5678원해요!
     즉시이동 https://www.kfo.ai/course/BIODATA_ORACLE.html!
     hi-d@learners-hi.co.kr 메일 환영!
     """

## 1. @ 외의 특수기호를 제외한 한글과 영어를 추출하는 정규표현식을 찾으세요
import re
num1 = re.findall(r'[가-힣a-zA-Z@]+', text) # 한글, 영어, @
print("한글/영어/@만 추출:", num1)

## 2. 모든 전화번호만 추출하는 정규표현식을 찾으세요
num2 = re.findall(r'\d{2,3}-\d{3,4}-\d{4}', text)
print("전화번호 추출:", num2)

## 3. 홈페이지 주소만 추출하는 정규표현식을 찾으세요
num3 = re.findall(r'https://[^\s!]+', text) # [^\s] 공백이 아닌 문자, !도 빼기
print("홈페이지 주소 : ", num3)

## 4. 이메일 주소만 추출하는 정규표현식
num4 = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-0.-]+\.[a-zA-Z]{2,}', text) # .은 아무문자라는 의미가 있어 \. 사용
print("이메일 : ", num4)

## 5. 훈련장렬금 얼마인지 추출하는 정규표현식
# num5 = re.findall(r'[0-9]+\만원', text)
num5 = re.findall(r'\d{1,3}만원', text) # \d가 1~3자리
print("훈련장려금 : ", num5)




#### 수업 추가 실습
pattern=r"\w+@\w+|\w+"
print(re.findall(pattern,text))

cleaned = re.sub(r'[🔥・♨♥☆§💸▶★]', '', text)           # 특수문자 제거
print(cleaned)   # 정제 결과 확인

cleaned = re.sub(r'[🔥・♨♥☆§💸]|(?<=[가-힣])/|(?<=[A-z])/', '', text)           # 특수문자 제거
print(cleaned)   # 정제 결과 확인

pattern=r"\d+-\d+-\d+"
print(re.findall(pattern,text))

pattern=r"\d{2,}-\d{3,4}-\d{4}"
print(re.findall(pattern,text))

pattern=r"https:\/\/[^\s가-힣!%@\\]+"
print(re.findall(pattern,text))

pattern=r"https:\/\/[a-zA-z\.\/]+"
print(re.findall(pattern,text))

pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-z]{2,}"
print(re.findall(pattern,text))

pattern=r"[\w-]+@[\w-]+\.[A-z]{2,}\.*[A-z]*"
print(re.findall(pattern,text))

pattern=r"\s(\d+\w원)"
print(re.findall(pattern,text))

pattern=r"\s(\d+[만|천]원)"
print(re.findall(pattern,text))
