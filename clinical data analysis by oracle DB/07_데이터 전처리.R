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


## 시간데이터 전처리_날짜 계산

install.packages("tidyverse")
library(tidyverse)

date1 <- ymd("2075-03-27")
date2 <- mdy("03/31/2075")
date2 - date1
year(date1)
month(date2)
wday(date2, label = T)
date2+years(7) # 연도 + 7년

# as.Date 사용해서 계산

head(as.Date(final_data$endtime) - as.Date(final_data$Adm))


# POSIXct 속성 이용해서 계산
head((as.numeric(final_data$endtime) - as.numeric(final_data$Adm))/3600/24) # numeric 전환 시 단위가 min -> 일 단위 차이

str(final_data$endtime)
str(final_data$Adm)

attr(final_data$endtime, "tzone")
attr(final_data$Adm, "tzone")


# difftime() 사용하기
difftime(final_data$endtime, final_data$Adm, units = "days") %>% head



## 결측치 처리
is.na(final_data$Op)
which(is.na(final_data$Op)) # 원소들의 위치 확인

colSums(is.na(final_data)) # 모든 컬럼의 결측치 확인

summary(final_data) # 요약 통계
