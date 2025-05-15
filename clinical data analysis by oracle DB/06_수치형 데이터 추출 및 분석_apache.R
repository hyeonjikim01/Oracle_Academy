#### APACHE2 score ####

## 데이터 불러오기
library(readxl)
raw <- read_excel("C:/Users/KDT_09/hjkim01_git/clinical data analysis by oracle DB/kdt_sepsis_crf_apache.xlsx")
str(raw)
df1 <- raw
library(dplyr)

raw2 <- read_excel("C:/Users/KDT_09/hjkim01_git/clinical data analysis by oracle DB/kdt_sepsis_crf_sofa.xlsx")
sofa <- raw2

df <- df1 %>%
  left_join(sofa %>% select(ID, GCS), by = "ID")

df <- df %>%
  	mutate(across(everything(), ~ as.numeric(.)))
str(df)
View(df)

colSums(is.na(df))

## 점수 부여

# Age
df$new_age <- ifelse(df$Age < 44, 0,
              ifelse(df$Age >= 45 & df$Age <= 54, 2,
              ifelse(df$Age >= 55 & df$Age <= 64, 3,
              ifelse(df$Age >= 65 & df$Age <= 74, 5,
              ifelse(df$Age >= 75, 6, NA)))))
table(df$new_age)

# GCS
sum(is.na(sofa$GCS))
df$new_gcs <- 15 - df$GCS


# MBP
df$new_mbp <- ifelse(df$MBP < 50, 4,
              ifelse(df$MBP >= 50 & df$MBP <= 69, 2,
              ifelse(df$MBP >= 70 & df$MBP <= 109, 0,
              ifelse(df$MBP >= 110 & df$MBP <= 129, 2,
              ifelse(df$MBP >= 130 & df$MBP <= 159, 3,
              ifelse(df$MBP >= 160, 4, NA))))))
table(df$new_mbp)

# HR 심박수
df$new_hr <-  ifelse(df$HR >= 0 & df$HR <= 39, 4,
              ifelse(df$HR >= 40 & df$HR <= 54, 3,
              ifelse(df$HR >= 55 & df$HR <= 69, 2,
              ifelse(df$HR >= 70 & df$HR <= 109, 0,
              ifelse(df$HR >= 110 & df$HR <= 139, 2,
              ifelse(df$HR >= 140 & df$HR <= 179, 3,
              ifelse(df$HR >= 180, 4, NA)))))))
table(df$new_hr)

# RR 호흡수
df$new_rr <-  ifelse(df$RR >= 0 & df$RR <= 5, 4,
              ifelse(df$RR >= 6 & df$RR <= 9, 2,
              ifelse(df$RR >= 10 & df$RR <= 11, 1,
              ifelse(df$RR >= 12 & df$RR <= 24, 0,
              ifelse(df$RR >= 25 & df$RR <= 34, 1,
              ifelse(df$RR >= 35 & df$RR <= 49, 3,
              ifelse(df$RR >= 50, 4, NA)))))))
table(df$new_rr)

# temp 체온
df$new_temp <- ifelse(df$Temp >= 41, 4,
               ifelse(df$Temp >= 39 & df$Temp < 41, 3,
               ifelse(df$Temp >= 38.5 & df$Temp < 39, 1,
               ifelse(df$Temp >= 36 & df$Temp < 38.5, 0,
               ifelse(df$Temp >= 34 & df$Temp < 36, 1,
               ifelse(df$Temp >= 32 & df$Temp < 34, 2,
               ifelse(df$Temp >= 30 & df$Temp < 32, 3,
               ifelse(df$Temp < 30, 4, NA))))))))
table(df$new_temp)

# pH 산도
df$new_ph <- ifelse(df$pH < 7.15, 4,
             ifelse(df$pH >= 7.15 & df$pH < 7.25, 3,
             ifelse(df$pH >= 7.25 & df$pH < 7.33, 2,
             ifelse(df$pH >= 7.33 & df$pH < 7.5, 0,
             ifelse(df$pH >= 7.5 & df$pH < 7.6, 1,
             ifelse(df$pH >= 7.6 & df$pH < 7.7, 2,
             ifelse(df$pH >= 7.7, 4, NA)))))))
table(df$new_ph)


# 호흡
df$new_resp <- ifelse(df$FiO2 >= 0.5, 
  # FiO₂ ≥ 0.5 → A-a gradient 사용
  ifelse(df$"A-a gradient" >= 500, 4,
  ifelse(df$"A-a gradient" >= 350 & df$"A-a gradient" < 500, 3,
  ifelse(df$"A-a gradient" >= 200 & df$"A-a gradient" < 350, 2,
  ifelse(df$"A-a gradient" < 200, 0, NA)))),
  
  # FiO₂ < 0.5 → PaO₂ 사용
  ifelse(df$PaO2 < 55, 4,
  ifelse(df$PaO2 >= 55 & df$PaO2 <= 60, 3,
  ifelse(df$PaO2 >= 61 & df$PaO2 <= 70, 1,
  ifelse(df$PaO2 > 70, 0, NA))))
)
table(df$new_resp)


# Na 혈중 나트륨
df$new_na <- ifelse(df$Na < 111, 4,
             ifelse(df$Na >= 111 & df$Na <= 119, 3,
             ifelse(df$Na >= 120 & df$Na <= 129, 2,
             ifelse(df$Na >= 130 & df$Na <= 149, 0,
             ifelse(df$Na >= 150 & df$Na <= 154, 1,
             ifelse(df$Na >= 155 & df$Na <= 159, 2,
             ifelse(df$Na >= 160 & df$Na <= 179, 3,
             ifelse(df$Na >= 180, 4, NA))))))))
table(df$new_na)

# K 혈중 칼륨
df$new_k <- ifelse(df$K < 2.5, 4,
            ifelse(df$K >= 2.5 & df$K < 3.0, 2,
            ifelse(df$K >= 3.0 & df$K < 3.5, 1,
            ifelse(df$K >= 3.5 & df$K <= 5.4, 0,
            ifelse(df$K >= 5.5 & df$K <= 5.9, 1,
            ifelse(df$K >= 6.0 & df$K <= 6.9, 3,
            ifelse(df$K >= 7.0, 4, NA)))))))
table(df$new_k)

# Cr 크레아티닌
df$new_cr <- (
  ifelse(df$Cr < 0.6, 2,
  ifelse(df$Cr >= 0.6 & df$Cr <= 1.4, 0,
  ifelse(df$Cr >= 1.5 & df$Cr <= 1.9, 2,
  ifelse(df$Cr >= 2.0 & df$Cr <= 3.4, 3,
  ifelse(df$Cr >= 3.5, 4, NA)))))
) * ifelse(df$ARF == 1, 2, 1)
table(df$new_cr)

# HCT 헤마토크릿
df$new_hct <- ifelse(df$HCT < 20, 4,
              ifelse(df$HCT >= 20 & df$HCT < 30, 2,
              ifelse(df$HCT >= 30 & df$HCT < 46, 0,
              ifelse(df$HCT >= 46 & df$HCT < 50, 1,
              ifelse(df$HCT >= 50 & df$HCT < 60, 2,
              ifelse(df$HCT >= 60, 4, NA))))))
table(df$new_hct)

# WBC 백혈구 수
df$new_wbc <- ifelse(df$WBC < 1, 4,
              ifelse(df$WBC >= 1 & df$WBC < 3, 2,
              ifelse(df$WBC >= 3 & df$WBC < 15, 0,
              ifelse(df$WBC >= 15 & df$WBC < 20, 1,
              ifelse(df$WBC >= 20 & df$WBC < 40, 2,
              ifelse(df$WBC >= 40, 4, NA))))))
table(df$new_wbc)

# Op 수술유형
table(df$Op)
df$new_op <- ifelse(df$Op == 1, 5,
              ifelse(df$Op == 2, 2,
              ifelse(df$Op == 3, 5, NA)))
table(df$new_op)
table(df$Op)


## APACHE2 점수 구하기
score_vars <- c("new_age", "new_mbp", "new_hr", "new_rr", "new_temp",
                "new_ph", "new_resp", "new_na", "new_k", "new_cr",
                "new_hct", "new_wbc", "new_op", "new_gcs")

df$newAPACHE <- rowSums(df[, score_vars], na.rm = TRUE)



## 새로운 점수와 기존 점수 차이 구하기
colnames(df)
df$apa_diff <- abs(df$newAPACHE - df$'APACHE II score')
diff2 <- subset(df, abs(df$newAPACHE - df$'APACHE II score') > 2)  # 2점 이상 차이나는 행만
View(diff2)

## 상과계수 구하기
score_vars <- c("new_age", "new_gcs", "new_mbp", "new_hr", "new_rr", "new_temp",
                "new_ph", "new_resp", "new_na", "new_k", "new_cr",
                "new_hct", "new_wbc", "new_op")

cor(df[, score_vars], df$apa_diff, use = "complete.obs") ## new_gcs와 new_resp이 높음

write.csv(df, "df1.csv", row.names = FALSE)

## gcs 점수 보정하기

# gcs 점수를 제거한 점수 계산
df$newAPACHE_wo_gcs <- df$newAPACHE - df$new_gcs

# 기존 점수에서 GCS 역산
df$estimated_gcs <- df$`APACHE II score` - df$newAPACHE_wo_gcs
df$estimated_gcs[df$estimated_gcs < 0] <- NA

# 보정된 GCS로 newAPACHE 다시 계산산
df$newAPACHE_adjusted <- df$newAPACHE_wo_gcs + df$estimated_gcs
df$adjusted_diff <- df$newAPACHE_adjusted - df$`APACHE II score`
df$within_2 <- abs(df$adjusted_diff) <= 2
table(df$within_2)
table(df$adjusted_diff)


###########################################################################################

## 4. 최종 데이터셋 구성

a <- select(sofa,
            ID, Adm, endtime, Age, Sex, Height, Weight,
            SBP, DBP, MBP, VASO, `mcg/kg/min`,
            AF_init, AF_hos, MV, GCS, `Urine output`,
            Plt, Bil, LACTATE, Cr, result, SOFA)


b <- select(df1,
            ID, Op, PaO2, FiO2, HR, RR, WBC, Na, K, pH, HCT, `APACHE II score`)

# 3. ID 기준으로 left join
final_data <- left_join(x = a, y = b, by = "ID")
View(final_data)
str(final_data)
write.csv(final_data, "final_data.csv", row.names = FALSE)
