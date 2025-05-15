## 패키지 불러오기

library(dplyr)


## 데이터

library(DBI)
library(odbc)
con <- dbConnect(odbc::odbc(), Driver = "Oracle in instantclient_23_8",
                 DBQ = "OracleDB", UID = "EDU119", PWD = "oracle_4U")
df1 <- dbGetQuery(con, "SELECT * FROM KDT_SEPSIS_CRF_SOFA")
df2 <- dbGetQuery(con, "SELECT * FROM KDT_SEPSIS_CRF_apache")

a <- select(df1,
            ID, Adm, endtime, Age, Sex, Height, Weight,
            SBP, DBP, MBP, VASO, `mcg/kg/min`,
            AF_init, AF_hos, MV, GCS, `Urine output`,
            Plt, Bil, LACTATE, Cr, result, SOFA)


b <- select(df2,
            ID, Op, PaO2, FiO2, HR, RR, WBC, Na, K, pH, HCT, `APACHE II score`)

final_data <- left_join(x = a, y = b, by = "ID")
str(final_data)
View(final_data)

#### 1. Demography 인구통계학적 분석

# 사망한 환자와 생존한 환자는 어떤 점이 다르며, 그 차이가 통계적으로 의미가 있는가

final_data$AF <- ifelse(final_data$AF_init == 1|final_data$AF_hos == 1, 1, 0)
final_data$Death <- ifelse(final_data$result == 0|final_data$result == 1, 0, 1)
final_data$Sex <- as.character(final_data$Sex)
final_dataVASO <- as.character(final_data$VASO)
final_data$AF <- as.character(final_data$AF)
final_dataMv <- as.character(final_data$MV)
final_data$Op <- as.character(final_data$Op)
table(final_data$Death)
write.csv(final_data, "final_data.csv", row.names = FALSE)


install.packages("tableone")
library(tableone)
figure1<- CreateTableOne(data = final_data,
                vars = c("Age", "Sex", "Height", "Weight", "SBP", "DBP", "MBP",
                         "VASO", "AF", "MV", "GCS", "Urine output", "Plt", "Bil",
                         "LACTATE", "Cr", "Op", "PaO2", "FiO2", "HR",
                         "RR", "WBC", "Na", "K", "pH", "HCT",
                         "SOFA", "APACHE II score"),
               strata = "Death")

figure1
print(figure1, showAllLevels = T)

write.csv(print(figure1, showAllLevels = T), "figure1.csv")


#### 2. SOFA 단일 모델

# SOFA 점수가 사망 여부(Death)를 얼마나 잘 예측하는지를 평가

install.packages("Epi")
library(Epi)

ROC(form = Death ~ SOFA, data = final_data, plot = "ROC", AUC = TRUE)

model1 <- glm(Death~SOFA, data = final_data, family = "binomial")
pred <- predict(model1, type = "response")
summary(model1)
