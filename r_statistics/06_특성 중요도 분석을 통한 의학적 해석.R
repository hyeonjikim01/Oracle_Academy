## 패키지 불러오기
library(dplyr)
library(caret)
install.packages("randomForest")
library(randomForest)
install.packages("xgboost")
library(xgboost)
install.packages("SHAPforxgboost")
library(SHAPforxgboost)
library(xgboost)
library(Matrix)
library(ggplot2)


# 시드 고정
set.seed(42)

# CSV 파일 불러오기
setwd("C://Users//KDT_09")
data <- read.csv("C:/heart_disease_uci.csv")

# target 변수 생성
data$target <- ifelse(data$num > 0, 1, 0)

# 결측치 및 빈 문자열 제거
library(dplyr)
data2 <- filter(data, if_all(everything(), ~ !is.na(.) & . != ""))

# 불필요한 변수 제거
data2sub <- subset(data2, select = c(-id, -num))

# 데이터 분할 (80% 학습용, 20% 테스트용)
library(caret)
train_index <- createDataPartition(data2sub$target, p = 0.8, list = FALSE)
train_data <- data2sub[train_index, ]
test_data  <- data2sub[-train_index, ]

# 로지스틱 회귀 모델 적합 + 변수 선택
logit_model <- glm(target ~ ., data = train_data, family = binomial)
summary(step(logit_model))


### 랜덤포레스트 ###
train_data$target <- as.factor(train_data$target)
rf_model <- randomForest(target ~ ., data = train_data, importance = TRUE, ntree = 500)
rf_importance <- importance(rf_model)
rf_importance

### XGboost ###
train_matrix <- model.matrix(target ~ . -1, data = train_data)
train_label <- as.numeric(as.character(train_data$target))
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

params <- list(
		   objective = "binary:logistic",
		   eval_metric = "logloss",
		   tree_method = "auto"
		   )

# 모델 학습
xgb_model <- xgb.train(params, dtrain, nrounds = 100, verbose = 0)

# 변수 중요도 확인
importance <- xgb.importance(model = xgb_model)
importance
xgb.plot.importance(importance)

# XGboost SHAP
shap_values <- predict(xgb_model, dtrain, predcontrib = TRUE)
shap_values <- as.data.frame(shap_values)
colnames(shap_values) <- c(colnames(train_matrix), "bias")
shap_values <- shap_values[, !colnames(shap_values) %in% "bias"]

shap_long <- shap.prep(shap_contrib = shap_values, X = as.data.frame(train_matrix))
shap.plot.summary(shap_long)
shap.plot.dependence(data_long = shap_long, x = "age")