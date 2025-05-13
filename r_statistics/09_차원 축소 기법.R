#### 인터랙티브 3D 시각화 ####

# 패키지
install.packages("plotly")
library(plotly)
library(MASS)

## 데이터
set.seed(42)
mean_vec <- c(0, 0, 0) # 3차원 벡터
cov_matrix <- matrix(c(3, 1, 1, # 공분산 행렬
			     1, 2, 1,
			     1, 1, 1), nrow = 3)
data3D <- mvrnorm(n = 100, mu = mean_vec, Sigma = cov_matrix) # 3차원 정규분포 따라 100개 샘플 생성

df3D <- as.data.frame(data3D) # 생성된 데이터 데이터프레임으로 변환
colnames(df3D) <- c("X1", "X2", "X3")

set.seed(42)
clusters <- kmeans(df3D[, c("X1", "X2", "X3")], centers = 3) # X1, X2, X3기준으로 K-Means 군집화 수행
df3D$cluster <- as.factor(clusters$cluster) # 군집 결과를 df3D에 추가

plot_ly(df3D, x = ~X1, y = ~X2, z = ~X3, color = ~cluster, # 각 축에 해당하는 변수
	  colors = c("red", "blue", "green"), type = "scatter3d", mode = "markers", # 3d 점 그래프
	  marker = list(size = 5)) %>%
	layout(title = "3D 데이터 (K-Means 군집 기반 색상 구분)")


#### 주성분 분석 실습 ####

## 패키지
library(ggplot2)
install.packages("FactoMineR")
library(FactoMineR)
install.packages("factoextra")
library(factoextra)
library(dplyr)


## 데이터
data("swiss")
df <- swiss
df %>% str
? swiss


## 주성분 분석 수행
pca_result <- prcomp(df, center = TRUE, scale. = TRUE)
summary(pca_result)
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 75))

# 기존 변수는 새로운 주성분들의 조합으로 표현 가능
pca_result$rotation

# 새로운 주성분들의 직교 확인
round(t(pca_result$rotation) %*% pca_result$rotation, 5)

# 파이프 함수로도 사용가능
t(pca_result$rotation) %*% pca_result$rotation %>% round(4)


## 시각화
# biplot
fviz_pca_biplot(pca_result, repel = TRUE, col.var = "red", col.ind = "blue")

# 변수 기여도 시각화
fviz_pca_var(pca_result, col.var = "contrib", gradient.cols = c("blue", "red"))

# 개별데이터 주성분 분포
fviz_pca_ind(pca_result, col.ind = "cos2", gradient.cols = c("blue", "red"))


#### 요인 분석 예제 실습 ####

install.packages("psych")
library(psych)
df <- mtcars
str(df)
which(is.na(df))

# kmo 검사
KMO(df)

# bartlett 구형성 검사
cortest.bartlett(df)


## 요인 분석 적 데이터 체크
eigen(cor(scale(df)))
eigen(cor(scale(df)))$value %>% plot
VSS.scree(scale(df))

## mtcars 사용
fa_result <- fa(df, nfactors = 2, rotate = "varimax", fm = "ml")
fa_result