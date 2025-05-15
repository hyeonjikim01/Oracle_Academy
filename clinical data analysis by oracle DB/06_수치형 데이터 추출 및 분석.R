#### 수치형 데이터 추출 및 분석 ####

## 패키지
library(dplyr)

## 데이터
data <- dbGetQuery(con, "SELECT * FROM KDT_SEPSIS_CRF_SOFA")
head(data)

library(readxl)
raw <- read_excel("C:/Users/KDT_09/hjkim01_git/clinical data analysis by oracle DB/kdt_sepsis_crf_sofa.xlsx")
str(raw)

## SOFA 점수 구하기
data$cc/hr %>% summary # 에러
data$"cc/hr" %>% summary

######################################################


## SOFA 점수 구하기

# respiratory
rep <- dbGetQuery(con, '
SELECT  id
	, "PaO2/FiO2"
	, mv
  	,  CASE 
    		WHEN "PaO2/FiO2" < 100 AND mv = 1 THEN 4
    		WHEN "PaO2/FiO2" < 100 AND mv = 0 THEN 2
    		WHEN "PaO2/FiO2" < 200 AND mv = 1 THEN 3
    		WHEN "PaO2/FiO2" < 200 AND mv = 0 THEN 2
    		WHEN "PaO2/FiO2" < 300 THEN 2
    		WHEN "PaO2/FiO2" < 400 THEN 1
    		ELSE 0
  	   END AS resp_score
FROM KDT_SEPSIS_CRF_SOFA
')
View(rep)


# Coagulation
coag <- dbGetQuery(con, '
SELECT  id
	, "Plt"
  	,  CASE 
    		WHEN "Plt" < 20 THEN 4
    		WHEN "Plt" <= 49 THEN 3
    		WHEN "Plt" <= 99 THEN 2
    		WHEN "Plt" <= 149 THEN 1
    		ELSE 0
  	   END AS plt_score
FROM KDT_SEPSIS_CRF_SOFA
')
View(coag)

# Liver
liv <- dbGetQuery(con, '
SELECT  id
  	, "Bil"
  	, "Urine output"
  	, CASE
    		WHEN "Bil" >= 5 OR "Urine output" < 200 THEN 4
    		WHEN "Bil" > 3.4 OR "Urine output" < 500 THEN 3
    		WHEN "Bil" > 1.9 THEN 2
    		WHEN "Bil" > 1.2 THEN 1
    		WHEN "Bil" <= 1.2 THEN 0
  	   END AS b_score
FROM KDT_SEPSIS_CRF_SOFA
')

View(liv)
head(liv)

# Cadiovascular
cadi <- dbGetQuery(con, '
SELECT  id
      , mbp
      , vaso       -- 0: 없음, 1: 도파민, 2: 도부타민, 3: 에피네프린, 4: 노르에피네프린
      , "VASO_dose"
      , CASE
            WHEN mbp >= 70 AND vaso = 0 THEN 0 -- 정상혈압 & 승압제 미사용
            WHEN mbp < 70 AND vaso = 0 THEN 1  -- 저혈압 & 승압제 미사용

            WHEN vaso = 1 AND "VASO_dose" <= 5 THEN 2  -- 도파민 ≤ 5
            WHEN vaso = 2 THEN 2                       -- 도부타민은 무조건 점수 2

            WHEN vaso = 1 AND "VASO_dose" > 5 AND "VASO_dose" <= 15 THEN 3  -- 도파민 > 5 ≤ 15
            WHEN vaso IN (3, 4) AND "VASO_dose" <= 0.1 THEN 3               -- 에피/노르에피네프린 ≤ 0.1

            WHEN vaso = 1 AND "VASO_dose" > 15 THEN 4                       -- 도파민 > 15
            WHEN vaso IN (3, 4) AND "VASO_dose" > 0.1 THEN 4                -- 에피/노르에피네프린 > 0.1

            ELSE NULL
        END AS vaso_score
FROM kdt_sepsis_crf_sofa
')

View(cadi)


#############################################################################################################################

# 1. 원본 데이터 복사
test <- raw

## 2. 각 점수 계산 (벡터 단위로 처리)

# respiratory
resp_score <- ifelse(test$`PaO2/FiO2` < 100 & test$MV == 1, 4,
              ifelse(test$`PaO2/FiO2` < 100 & test$MV == 0, 2,
              ifelse(test$`PaO2/FiO2` < 200 & test$MV == 1, 3,
              ifelse(test$`PaO2/FiO2` < 200 & test$MV == 0, 2,
              ifelse(test$`PaO2/FiO2` < 300, 2,
              ifelse(test$`PaO2/FiO2` < 400, 1, 0))))))

# Coagulation
plt_score <- ifelse(test$Plt < 20, 4,
             ifelse(test$Plt <= 49, 3,
             ifelse(test$Plt <= 99, 2,
             ifelse(test$Plt <= 149, 1, 0))))

# Liver
b_score <- ifelse(test$Bil >= 5 | test$`Urine output` < 200, 4,
           ifelse(test$Bil > 3.4 | test$`Urine output` < 500, 3,
           ifelse(test$Bil > 1.9, 2,
           ifelse(test$Bil > 1.2, 1, 0))))

# Cadiovascular
vaso_score <- ifelse(test$MBP >= 70 & test$VASO == 0, 0,
              ifelse(test$MBP < 70 & test$VASO == 0, 1,
              ifelse(test$VASO == 1 & test$VASO_dose <= 5, 2,
              ifelse(test$VASO == 2, 2,
              ifelse(test$VASO == 1 & test$VASO_dose > 5 & test$VASO_dose <= 15, 3,
              ifelse(test$VASO %in% c(3, 4) & test$VASO_dose <= 0.1, 3,
              ifelse(test$VASO == 1 & test$VASO_dose > 15, 4,
              ifelse(test$VASO %in% c(3, 4) & test$VASO_dose > 0.1, 4, NA))))))))

# Renal
renal_score <- ifelse(test$Cr >= 5.0 | test$`Urine output` < 200, 4,
               ifelse((test$Cr >= 3.5 & test$Cr <= 4.9) | test$`Urine output` < 500, 3,
               ifelse(test$Cr >= 2.0 & test$Cr <= 3.4, 2,
               ifelse(test$Cr >= 1.2 & test$Cr <= 1.9, 1,
               ifelse(test$Cr < 1.2, 0, NA)))))

# CNS
gcs_score <- ifelse(test$GCS == 15, 0,
             ifelse(test$GCS >= 13, 1,
             ifelse(test$GCS >= 10, 2,
             ifelse(test$GCS >= 6, 3,
             ifelse(test$GCS < 6, 4, NA)))))


# 3. 새 테이블 생성
new_sofa <- data.frame(
  id = test$ID,
  resp_score = resp_score,
  plt_score = plt_score,
  b_score = b_score,
  vaso_score = vaso_score,
  renal_score = renal_score,
  gcs_score = gcs_score,
  sofa = test$SOFA
)


# 4. 총점 계산
new_sofa$'newSOFA' <- rowSums(new_sofa[, c("resp_score", "plt_score", "b_score", "vaso_score", "renal_score", "gcs_score")], na.rm = TRUE)
new_sofa <- subset(new_sofa, select = -total_sofa)

# 5. 기존 sofa와 점수 차이 구하기
new_sofa$diff <- new_sofa$'newSOFA'- new_sofa$sofa # 에러
str(new_sofa)
new_sofa$diff <- as.numeric(new_sofa$newSOFA) - as.numeric(new_sofa$sofa)

# 6. 결과 확인
head(new_sofa)
View(new_sofa)


## 기존 sofa와 새로운 sofa가 차이나는 원인 분석하고 오차를 2점 내로 줄이기
