library(readxl)
raw <- read_excel("C:/Users/KDT_09/hjkim01_git/clinical data analysis by oracle DB/kdt_sepsis_crf_sofa.xlsx")
str(raw)

test <- raw
str(test)


## 2. 각 점수 계산
# resliratory
test$PFscore <- 0
test$"PaO2/FiO2" <- as.numeric(test$"PaO2/FiO2")
test$"MV" <- as.numeric(test$"MV")

test$PFscore <- ifelse(test$`PaO2/FiO2` < 400, 1, test$PFscore)
test$PFscore <- ifelse(test$`PaO2/FiO2` < 300, 2, test$PFscore)
test$PFscore <- ifelse(test$`PaO2/FiO2` < 200 & test$MV == 1, 3, test$PFscore)
test$PFscore <- ifelse(test$`PaO2/FiO2` < 100 & test$MV == 1, 4, test$PFscore)
table(test$PFscore)

# Coagulation
test$"Plt" <- as.numeric(test$"Plt")
test$plt_score <- ifelse(test$Plt < 20, 4,
            	ifelse(test$Plt <= 49, 3,
             	ifelse(test$Plt <= 99, 2,
             	ifelse(test$Plt <= 149, 1, 0))))

# Liver
test$"Bil" <- as.numeric(test$"Bil")
test$"Urine output" <- as.numeric(test$"Urine output")
test$b_score <- ifelse(test$Bil >= 5 | test$`Urine output` < 200, 4,
           ifelse(test$Bil > 3.4 | test$`Urine output` < 500, 3,
           ifelse(test$Bil > 1.9, 2,
           ifelse(test$Bil > 1.2, 1, 0))))


# Cadiovascular
test$"MBP" <- as.numeric(test$"MBP")
test$VASO <- as.numeric(test$VASO)
test$`mcg/kg/min` <- as.numeric(test$`mcg/kg/min`)


test$vaso_score <- ifelse(test$MBP >= 70 & test$VASO == 0, 0,
              ifelse(test$MBP < 70 & test$VASO == 0, 1,

              ifelse(
                 (test$VASO == 1 & test$`mcg/kg/min` <= 5) |
                 (test$VASO == 2 & test$`mcg/kg/min` <= 5), 2,

              ifelse(
                 (test$VASO == 1 & test$`mcg/kg/min` > 5 & test$`mcg/kg/min` <= 15) |
                 (test$VASO == 2 & test$`mcg/kg/min` > 5 & test$`mcg/kg/min` <= 15) |
                 (test$VASO == 3 & test$`mcg/kg/min` <= 0.1) |
                 (test$VASO == 4 & test$`mcg/kg/min` <= 0.1), 3,

              ifelse(
                 (test$VASO == 1 & test$`mcg/kg/min` > 15) |
                 (test$VASO == 2 & test$`mcg/kg/min` > 15) |
                 (test$VASO == 3 & test$`mcg/kg/min` > 0.1) |
                 (test$VASO == 4 & test$`mcg/kg/min` > 0.1), 4, NA)))))


# Renal
test$Cr <- as.numeric(test$Cr)

test$renal_score <- ifelse(test$Cr >= 5.0 | test$`Urine output` < 200, 4,
               ifelse((test$Cr >= 3.5 & test$Cr <= 4.9) | test$`Urine output` < 500, 3,
               ifelse(test$Cr >= 2.0 & test$Cr <= 3.4, 2,
               ifelse(test$Cr >= 1.2 & test$Cr <= 1.9, 1,
               ifelse(test$Cr < 1.2, 0, NA)))))



# CNS
test$GCS <- as.numeric(test$GCS)

test$gcs_score <- ifelse(test$GCS == 15, 0,
             ifelse(test$GCS >= 13, 1,
             ifelse(test$GCS >= 10, 2,
             ifelse(test$GCS >= 6, 3,
             ifelse(test$GCS < 6, 4, NA)))))

View(test)

## 3. 새 테이블 생성
new_sofa <- data.frame(
  id = test$ID,
  PFscore = test$PFscore,
  plt_score = test$plt_score,
  b_score = test$b_score,
  vaso_score = test$vaso_score,
  renal_score = test$renal_score,
  gcs_score = test$gcs_score,
  sofa = test$SOFA
)

View(new_sofa)

## 4. 총점 계산
new_sofa$'newSOFA' <- rowSums(new_sofa[, c("PFscore", "plt_score", "b_score", "vaso_score", "renal_score", "gcs_score")], na.rm = TRUE)



# 기존 sofa와 새로운 sofa 차이

str(new_sofa)
new_sofa$sofa <- as.numeric(new_sofa$sofa)
new_sofa$diff <- abs(new_sofa$newSOFA - new_sofa$sofa)

# 차이가 2점보다 더 많이 나는 것
diff2 <- subset(new_sofa, abs(new_sofa$newSOFA - new_sofa$sofa) > 2)  # 2점 이상 차이나는 행만
View(diff2)

# 차이가 2점 보다 많은 것에서 큰 순서대로 나열
diff2_order <- diff2[order(-diff2$diff), ]
View(diff2_order)

library(dplyr)

merged_data <- left_join(diff2_order, test, by = c("id" = "ID"))
View(merged_data)
merged_data$plt_diff <- merged_data$Plt - merged_data$plt_score


## 오차 줄이기
colSums(is.na(new_sofa))

# Step 1: 결측치 보정
new_sofa[is.na(new_sofa)] <- 0

# Step 2: 재계산
new_sofa$newSOFA <- rowSums(new_sofa[, c("PFscore", "plt_score", "b_score", "vaso_score", "renal_score", "gcs_score")])

# Step 3: 오차 확인
new_sofa$diff <- abs(new_sofa$newSOFA - new_sofa$sofa)

# Step 4: 필터링
table(new_sofa$diff > 2)  # TRUE가 줄어들었는지 확인

