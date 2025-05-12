# 패키지 불러오기
library(pROC)

p <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
r <- c(0, 1, 0, 1, 0, 1, 1, 1, 1)
dat1 <- data.frame(cbind(p, r))
auc(dat1$r, dat1$p)
plot(roc(dat1$r, dat1$p))
abline(v=0)
abline(h=0)

auc(dat1$r, dat1$p)



##### ROC곡선과 AUC #####

### 검증 데이터를 통한 모형 적합
data <- read.csv("C:/heart_disease_uci.csv")
data$target <- ifelse(data$num > 0, 1, 0)
data2 <- filter(data, if_all(everything(), ~!is.na(.) & . != "")) # na 아니고 빈 문자열이 아닌 경우에만 TRUE
data2sub <- subset(data2, select = c(-id, -num, -dataset))  # dataset 변수 제거

set.seed(42) # 무작위 분할 실행될 때마다 같은 결과 나오도록 설정
train_index <- createDataPartition(data2sub$target, p = 0.8, list = FALSE) # train, test로 분리
train_data <- data2sub[train_index, ]
test_data <- data2sub[-train_index, ]

## 로지스틱 회귀
logit_model <- glm(target ~ ., data = train_data, family = binomial)	# 이진 분류용 로지스틱 회귀모델 만들기
	# target ~ . 종속변수 target을 나머지 모든 변수를 사용해서 예측하겠다는 의미
logit_pred <- predict(logit_model, newdata = test_data, type = "response")  # 예측값을 확률형태로 반환

## 랜덤 포레스트
train_data$target <- as.factor(train_data$target) # 타겟 변수를 범주형으로 변환
rf_model <- randomForest(target ~ ., data = train_data, importance = TRUE, ntree = 500) # 변수 중요도 측정
rf_pred <- predict(rf_model, newdata = test_data, type = "prob")[, 2]  # 예측값을 확률형태로 반환

## XGBoost

# 모든 데이터에서 feature matrix를 만들 기준 공식 저장
formula <- as.formula(target ~ .) # target을 예측하기 위해 모든 변수를 사용

# 학습용 feature matrix (동일한 열 순서 및 구조 보장)
train_matrix <- model.matrix(formula, data = train_data)[, -1]  # 첫번째 열 intercept 제거
train_label <- as.numeric(as.character(train_data$target))

# 테스트용 feature matrix (학습과 같은 구조로 보장)
test_matrix <- model.matrix(formula, data = test_data)[, -1]
test_label <- as.numeric(as.character(test_data$target)) # 숫자형 벡터로 변환

# DMatrix 생성 (학습용/테스트용)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# 파라미터 설정
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  tree_method = "auto"
)

# 모델 학습
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)

# 예측
xgb_pred <- predict(xgb_model, dtest) # 에러

## 에러 수정
test_data$restecg <- factor(test_data$restecg, levels = levels(factor(train_data$restecg)))
table(test_data$restecg)

# 'dataset' 변수를 제외한 test_data로부터 feature matrix 생성 (intercept 제거)
test_matrix <- model.matrix(target ~ . - 1, data = select(test_data, c(-dataset)))

# 'dataset' 변수를 제외한 후 target 열만 추출하여 숫자형으로 변환 (예: 0, 1)
test_label <- as.numeric(as.character(select(test_data, c(-dataset)$target))

# XGBoost용 DMatrix 객체 생성
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# 학습된 XGBoost 모델을 사용하여 예측값 생성
xgb_pred <- predict(xgb_model, dtest)



## confusion matrix와 auc 구하기
logit_auc <- roc(test_data$target, logit_pred)
logit_pred_target <- ifelse(logit_pred > 0.5, 1, 0)
table(test_data$target, logit_pred_target)
auc(logit_auc)

rf_auc <- roc(test_data$target, rf_pred)
rf_pred_target <- ifelse(rf_pred > 0.5, 1, 0)
table(test_data$target, rf_pred_target)
auc(rf_auc)

xgb_auc <- roc(test_data$target, xgb_pred)
xgb_pred_target <- ifelse(xgb_pred > 0.5, 1, 0)
table(test_data$target, xgb_pred_target)
auc(xgb_auc)


## ROC 곡선 그리기
plot(logit_auc, col = "blue", main = "ROC Curves")
legend("bottomright", legend = c("Logistic"), col = c("blue"), lwd = 2)

plot(rf_auc, col = "red", main = "ROC Curves")
legend("bottomright", legend = c("Random Forest"), col = c("red"), lwd = 2)

plot(xgb_auc, col = "green", main = "ROC Curves")
legend("bottomright", legend = c("XGBoost"), col = c("green"), lwd = 2)


# 세 개의 그래프 하나의 플롯 안에서 보여주기
# Set up the plotting area to display 3 plots in a row
par(mfrow = c(1, 3))

# Plot each ROC curve
plot(logit_auc, col = "blue", main = "Logistic", lwd = 2)
plot(rf_auc, col = "red", main = "Random Forest", lwd = 2)
plot(xgb_auc, col = "green", main = "XGBoost", lwd = 2)

# Add legends for each plot
legend("bottomright", legend = c("Logistic"), col = c("blue"), lwd = 2)
legend("bottomright", legend = c("Random Forest"), col = c("red"), lwd = 2)
legend("bottomright", legend = c("XGBoost"), col = c("green"), lwd = 2)

# 세 그래프 한 번에 그리기
plot(logit_auc, col = "blue", main = "ROC Curves", lwd = 2)
plot(rf_auc, col = "red", add = TRUE)
plot(xgb_auc, col = "green", add = TRUE)
legend("bottomright", legend = c("Logistic", "Random Forest", "XGBoost"), col = c("blue", "red", "green"), lwd = 2)


#### 검증 데이터를 통한 모형 적합

TN <- table(test_data$target, logit_pred_target)[1, 1]
FP <- table(test_data$target, logit_pred_target)[1, 2]
FN <- table(test_data$target, logit_pred_target)[2, 1]
TP <- table(test_data$target, logit_pred_target)[2, 2]

accuracy <- (TP + TN) / (TN + FP + FN + TP)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)

print(c(accuracy, sensitivity, specificity, precision, f1_score))


# logit_pred는 threshold별로 accuracy, sensitivity, specificity, precision, fl_score 등을 저장할 빈 표
threshold <- seq(0.1, 0.9, by = 0.1)
logit_result <- matrix(NA, length(threshold), 6)
colnames(logit_result) <- c("threshold", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1_score")
logit_result <- data.frame(logit_result)


# logit_pred에 대한 성능 지표를 logit_result에 집어넣기
for ( i in 1:length(threshold))
	{
	logit_pred_target <- ifelse(logit_pred > threshold[i], 1, 0)
	TN <- table(test_data$target, logit_pred_target)[1, 1]
	FP <- table(test_data$target, logit_pred_target)[1, 2]
	FN <- table(test_data$target, logit_pred_target)[2, 1]
	TP <- table(test_data$target, logit_pred_target)[2, 2]

	accuracy <- (TP + TN) / (TN + FP + FN + TP)
	sensitivity <- TP / (TP + FN)
	specificity <- TN / (TN + FP)
	precision <- TP / (TP + FP)
	f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
	
	logit_result$threshold[i] <- threshold[i]
	logit_result$Accuracy[i] <- accuracy
	logit_result$Sensitivity[i] <- sensitivity
	logit_result$Specificity[i] <- specificity
	logit_result$Precision[i] <- precision
	logit_result$F1_score[i] <- f1_score
	}
	
	logit_result

# rf_pred에 대한 성능지표
threshold <- seq(0.1, 0.9, by = 0.1)
rf_result <- matrix(NA, length(threshold), 6)
colnames(logit_result) <- c("threshold", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1_score")
rf_result <- data.frame(rf_result)

for (i in 1:length(threshold)) {
  rf_pred_target <- ifelse(rf_pred > threshold[i], 1, 0)

  TN <- table(test_data$target, rf_pred_target)[1, 1]
  FP <- table(test_data$target, rf_pred_target)[1, 2]
  FN <- table(test_data$target, rf_pred_target)[2, 1]
  TP <- table(test_data$target, rf_pred_target)[2, 2]

  accuracy <- (TP + TN) / (TN + FP + FN + TP)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)

  rf_result$threshold[i]   <- threshold[i]
  rf_result$Accuracy[i]    <- accuracy
  rf_result$Sensitivity[i] <- sensitivity
  rf_result$Specificity[i] <- specificity
  rf_result$Precision[i]   <- precision
  rf_result$F1_score[i]    <- f1_score
}	
  rf_result


# xgb_result에 대한 성능지표
threshold <- seq(0.1, 0.9, by = 0.1)  # threshold 값 생성 (이미 있다면 생략 가능)
xgb_result <- matrix(NA, length(threshold), 6)  # NA로 초기화된 9행 6열 행렬 생성
colnames(xgb_result) <- c("threshold", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1_score")  # 열 이름 지정
xgb_result <- data.frame(xgb_result)  # 행렬을 데이터프레임으로 변환

for (i in 1:length(threshold)) {
  xgb_pred_target <- ifelse(xgb_pred > threshold[i], 1, 0)

  TN <- table(test_data$target, xgb_pred_target)[1, 1]
  FP <- table(test_data$target, xgb_pred_target)[1, 2]
  FN <- table(test_data$target, xgb_pred_target)[2, 1]
  TP <- table(test_data$target, xgb_pred_target)[2, 2]

  accuracy <- (TP + TN) / (TN + FP + FN + TP)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)

  xgb_result$threshold[i]   <- threshold[i]
  xgb_result$Accuracy[i]    <- accuracy
  xgb_result$Sensitivity[i] <- sensitivity
  xgb_result$Specificity[i] <- specificity
  xgb_result$Precision[i]   <- precision
  xgb_result$F1_score[i]    <- f1_score
}
  xgb_result




