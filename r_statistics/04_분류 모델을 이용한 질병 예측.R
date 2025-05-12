# package
install.packages("readxl")
library(readxl)
install.packages("ggplot2")
library(ggplot2)

## 데이터 불러오기
setwd("C://Users//KDT_09")
data <- read.csv("C:/heart_disease_uci.csv")
View(data)
str(data)


## 데이터 탐색
table(data$sex)
barplot(table(data$restecg))


## 데이터 전처리

# 결측치 처리
is.na(data)
dim(data) # 행, 열
data1 <- na.omit(data) # 결측치 제거
dim(data1)
data2 <- filter(data, if_all(everything(), ~!is.na(.)&.!=""))# na가 아닌 것, blank가 아닌 것
dim(data2)
data3 <- filter(data, data$sex == "Male") # 원하는 조건대로 뽑을 수도 있음
dim(data3)

# 변수 변환
data2$num

data2$target <- ifelse(data2$num >0, 1, 0) #새로운 변수에 저장
table(data2$target) # 빈도수 확인


## 모형적합
model <- glm(target ~ age+sex+dataset+cp+trestbps+chol+fbs+restecg+thalch+exang+oldpeak+slope+ca+thal , family = binomial, data = data2)
summary(model)

# data2에서 id와 num 변수를 제거한 새로운 데이터프레임 생성
data2sub <- subset(data2, select = c(-id, -num))
# 변수명 확인
colnames(data2sub)
# target을 종속변수로 하고 나머지 모든 변수(.)를 독립변수로 설정한 로지스틱 회귀모형 생성
model2 <- glm(target ~ ., family = binomial, data = data2sub)
# 모형 요약 출력
summary(model2)

summary(step(model2, direction = "both", k = 2))


## 변수 중요도 분석
install.packages("caret")
library(caret)

varImp(step(model2, direction = "both"))
varImp(model2)

vi <- varImp(model2)
barplot(names.arg = rownames(vi), vi$Overall, cex = 0.7)

# 변수 중요도 그래프
var_imp_df <- data.frame(Variable = rownames(vi), Importance = vi$Overall)
ggplot(var_imp_df, aes(x = Variable, y = Importance))+
coord_flip()+
geom_bar(stat = "identity")+
theme_minimal()


## train set & test set
set.seed(42)

train_index <- createDataPartition(data2sub$target, p = 0.7, list = FALSE) # 70% 훈련세트
train_data <- data2sub[train_index, ]
test_data <- data2sub[-train_index, ]

dim(data2sub)
dim(train_data)
dim(test_data)

## 훈련 데이터셋 로지스틱 회귀 모형 적합
train_model <- glm(target ~., family = "binomial", data = train_data)
summary(step(train_model))

## 로지스틱 회귀분석에서 다중공선성
install.packages("car")
library(car)

vif(step(train_model))

## 적합된 모형을 검증 데이터에 적용
library(caret)

# 예측 확률 → 예측 클래스
test_data$prob <- predict(step(train_model), test_data, type = "response")
test_data$pred <- ifelse(test_data$prob > 0.5, 1, 0)

# factor로 변환 + levels 강제 지정
test_data$pred <- factor(test_data$pred, levels = c(0, 1))
test_data$target <- factor(test_data$target, levels = c(0, 1))

# 혼동 행렬 출력
confusionMatrix(test_data$pred, test_data$target)
