# 선형회귀 히스토그램
x <- c(168, 160, 170, 158, 176, 161, 180, 183, 180, 167, 179, 171, 166)
y <- c(179, 169, 180, 160, 178, 170, 183, 187, 179, 172, 181, 173, 165)

hist(x, freq = F)
lines(density(x), col = "red", lwd = 2)



x <- c(168, 160, 170, 158, 176, 161, 180, 183, 180, 167, 179, 171, 166)
y <- c(179, 169, 180, 160, 178, 170, 183, 187, 179, 172, 181, 173, 165)

length(x)
length(y)

lm(y~x)
summary(lm(y~x))# 모형 요약


# 실습
install.packages("dplyr")
library(dplyr)

par(mfrow = c(2, 1))

x <- c(1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9)
y <- c(0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10)

plot(x, y, ylim = c(-6, 16))
abline(lm(y~x), col = "red")
lm(y~x) %>% summary
lm(y~x) %>% aov %>% summary

u <- c(1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9)
v <- c(-6, 1, 8, -4, 3, 10, -2, 5, 12, 0, 7, 14, 2, 9, 16)

plot(u, v, ylim = c(-6, 16))
abline(lm(v~u), col = "red")
lm(v~u) %>% summary
lm(v~u) %>% aov %>% summary

# 변수변환
set.seed(1)

x <- seq(-3, 4, by = 0.1)
y <- 3+x^2+rnorm(length(x), 0, 1)

plot(x, y)

lm(y~x)
summary(lm(y~x))



#################################

# 다중회귀분석

install.packages("writexl")
library(writexl)

write_xlsx(data.frame(cor(mtcars)), "C:\\cor.xlsx")

str(mtcars)

colnames(mtcars)

model2 <- lm(mpg ~. ,data = mtcars)
summary(model2)

par(mfrow = c(2, 2))
plot(model2)