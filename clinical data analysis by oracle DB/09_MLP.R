#### 전처리

# 패키지 로드
install.packages("RSNNS")
library(RSNNS)
library(pROC)
library(dplyr)

# 시드 고정 (재현성 확보)
set.seed(42)

# 데이터 복사 및 변수명 정제
mlp_data <- final_data
colnames(mlp_data) <- make.names(colnames(mlp_data))  # 변수명에 공백/특수문자 제거

# MLP 학습에 필요 없는 변수 제거
mlp_data <- select(mlp_data, -c(ID, Adm, endtime, result))

# factor/character형 변수를 numeric으로 변환
mlp_data$Sex  <- as.numeric(mlp_data$Sex)
mlp_data$VASO <- as.numeric(mlp_data$VASO)
mlp_data$MV   <- as.numeric(mlp_data$MV)
mlp_data$Op   <- as.numeric(mlp_data$Op)
mlp_data$AF   <- as.numeric(mlp_data$AF)
# mlp_data$time <- as.numeric(mlp_data$time)

mlp_data$Op_1 <- ifelse(mlp_data$Op == 1, 1, 0)
mlp_data$Op_2 <- ifelse(mlp_data$Op == 2, 1, 0)
mlp_data$Op_3 <- ifelse(mlp_data$Op == 3, 1, 0)
mlp_data$VASO_1 <- ifelse(mlp_data$VASO == 1, 1, 0)
mlp_data$VASO_2 <- ifelse(mlp_data$VASO == 2, 1, 0)
mlp_data$VASO_3 <- ifelse(mlp_data$VASO == 3, 1, 0)
mlp_data$VASO_4 <- ifelse(mlp_data$VASO == 4, 1, 0)
mlp_data <- select(mlp_data, -VASO)
mlp_data <- select(mlp_data, -Op)

chr_vars <- c('Sex', 'MV', 'AF', 'Op_1', 'Op_2', 'Op_3', 'VASO_1', 'VASO_2', 'VASO_3', 'VASO_4')

mlp_data2 <- cbind(scale(select(mlp_data, -c(chr_vars, Death))),
                         select(mlp_data, c(chr_vars)),
                         select(mlp_data, Death))

x <- as.matrix(select(mlp_data2, -Death))
y <- as.numeric(mlp_data2$Death)
which(is.na(x))

# 스필릿 세팅
idx <- sample(1:nrow(x), 0.8 * nrow(x))
train_x <- x[idx, ]
train_y <- decodeClassLabels(y[idx])
test_x <- x[-idx, ]
test_y <- y[-idx]


## MLP 수행
model <- mlp(train_x, train_y,
             size = 3,
             learnFuncParams = c(0.1), # 학습률 0.1
             maxit = 300, # 최대 반복수 300회
             linOut = FALSE) # 출력층 형식 -> FLASE는분류(이진형)

colSums(is.na(train_x))  # 각 열별 NA 개수 확인


## MLP의 ROC 확인
pred_train <- predict(model, train_x)[, 2]
roc_train <- roc(y[idx], pred_train)
cat("학습 데이터 AUC : ", auc(roc_train), "\n")

pred_test <- predict(model, test_x)[, 2]
roc_test <- roc(test_y, pred_test)
cat("테스트 데이터 AUC : ", auc(roc_test), "\n")


## for문을 사용한 MLP 은닉 노드 수 그리드 써치

# 1. size 그리드 정의 (은닉노드 수)
size_grid <- seq(5, 20)

# 2. 결과 저장용 데이터프레임 초기화
results <- data.frame(size = integer(), AUC = numeric())

# 3. 반복하면서 size 값을 바꿔가며 MLP 학습
for (s in size_grid) {
  model <- mlp(train_x, train_y,
               size = s,
               learnFuncParams = c(0.1),  # 학습률
               maxit = 500,
               linOut = FALSE)  # 분류 문제이므로 linOut = FALSE
  
  # 예측 수행
  pred <- predict(model, test_x)[, 2]  # 두 번째 열: 양성 클래스 확률
  
  # AUC 계산
  auc_value <- auc(test_y, pred)
  
  # 결과 저장
  results <- rbind(results, data.frame(size = s, AUC = auc_value))
}

# 4. 결과 출력
print(results)


## for문을 사용한 MLP 은닉 노드 수 + 학습률 그리드 써치

# 1. 그리드 설정
size_grid <- seq(5, 10)
lr_grid <- c(0.01, 0.05, 0.1, 0.2)

# 2. 결과 저장용 데이터프레임 초기화
results <- data.frame(size = integer(), lr = numeric(), AUC = numeric())

# 3. 이중 for문으로 모델 튜닝
for (s in size_grid) {
  for (lr in lr_grid) {
    model <- mlp(train_x, train_y,
                 size = s,
                 learnFuncParams = c(lr),
                 maxit = 500,
                 linOut = FALSE)
    
    # 예측 및 AUC 계산
    pred <- predict(model, test_x)[, 2]
    auc_value <- auc(test_y, pred)
    
    # 결과 저장
    results <- rbind(results, data.frame(size = s, lr = lr, AUC = auc_value))
  }
}

# 4. 결과 출력
print(results)


## 실습) 은닉 노드 수 + 학습률 + 반복 횟수 튜닝

# 하이퍼파라미터 그리드 정의
size_grid <- seq(5, 10)  # 첫 번째 은닉층 노드 수
lr_grid <- c(0.01, 0.05, 0.1, 0.2)  # 학습률
iter <- c(300, 500, 700)  # 반복 횟수 (maxit)

# 결과 저장용 데이터프레임
results <- data.frame(size = integer(), lr = numeric(), iter = integer(), AUC = numeric())

# 3중 for문: 각 조합에 대해 MLP 학습 및 평가
for (i in iter) {
  for (s in size_grid) {
    for (lr in lr_grid) {
      
      # 모델 학습: 은닉층 구조 c(s, 5, 3)
      model <- mlp(train_x, train_y,
                   size = c(s, 5, 3),
                   learnFuncParams = lr,
                   maxit = i,
                   linOut = FALSE)
      
      # 예측 및 AUC 계산
      pred <- predict(model, test_x)[, 2]
      auc_value <- auc(test_y, pred)
      
      # 결과 저장
      results <- rbind(results, data.frame(size = s, lr = lr, iter = i, AUC = auc_value))
    }
  }
}

# 전체 결과 출력
print(results)

# AUC가 가장 높은 조합 출력
best_auc <- results[which.max(results$AUC), ]
print(best_auc)

# 최적 조합으로 MLP 모델 학습
best_model <- mlp(train_x, train_y,
                  size = c(best_auc$size, 3),         # 은닉층 구조: best + 고정
                  learnFuncParams = best_auc$lr,      # 학습률
                  maxit = best_auc$iter,              # 반복 횟수
                  linOut = FALSE)

# 테스트셋 예측 (확률)
pred_prob <- predict(best_model, test_x)[, 2]

# 확률 → 분류값 (cutoff = 0.5 기준)
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# factor로 변환
pred_factor <- factor(pred_class, levels = c(0, 1))
true_factor <- factor(test_y, levels = c(0, 1))

# confusion matrix 생성
library(caret)
conf <- confusionMatrix(pred_factor, true_factor, positive = "1")

# 결과 출력
print(conf)
