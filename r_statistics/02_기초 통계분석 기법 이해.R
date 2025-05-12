x <- 10
y <- as.integer(10)
z <- FALSE
w <- "Hello, World!"
f <- factor(c("one", "two", "three"))

v1 <- c(1, 2, 3, 4, 5)
v2 <- c("a", "b", "c")
v3 <- c(TRUE, FALSE, TRUE)
v4 <- c(1, 2, "C", FALSE)
v4

v4 <- as.integer(v4)
v4

x <- c(1, 2, 3)
y <- c(4, 5, 6)

x+y
x*2
x^2
sum(y)

# 리스트
list2 <- list(name = "R", age = 25, scores = c(90, 85, 80))
str(list2)
list2$name
list2$age
list2$scores

# 행렬
1:9
seq(1, 9)
mat2 <- matrix(1:9, nrow = 3)
mat3 <- matrix(seq(1, 9), nrow = 3, byrow = T)
mat2
mat3
seq(1, 9, by = 2)

# 데이터프레임
data2 <- data.frame(name = c("A", "B"), score = c(30, 40))
str(data2)
data2

# 조건문
x <- 4
if (x > 5) {
	print("x는 5 이상이다")}

x <- 6
if (x > 5) {
	print("x는 5 이상이다")}

x <- 4.999
if (x >= 5) {
	print("x는 5 이상이다")
} else {
	print("x는 5 미만이다")
}


k <- c(1, 3, 5, 7, 9)
for (i in k)
{
print(i)
}



i <- 1

while(i <= 20) {
  if(i %% 2 == 0) {
    print(i)
  }
  i <- i + 1
}


# 사용자 정의 함수
even_number <- function(a) {
	if (a %% 2 == 0) {
	return("Yes")
	} else { return("No") }
}

even_number(2)
even_number(3)


# 카이제곱 검정

# 적합성 검정
chisq.test(c(89, 41, 22, 8), p = c(9/16, 3/16, 3/16, 1/16))

# 독립성 검정
chisq.test(matrix(c(8, 4, 2, 6), nrow = 2), correct = F)

# t분포
x <- c(31, 27, 35, 35, 32, 35, 29, 30, 33, 37, 30, 28, 34, 33, 28, 36, 33, 30)
t.test(x, mu = 30)

y <- c(28, 32, 37, 35, 28, 23, 35, 40, 33, 41, 35, 33, 31, 33, 33, 28, 35)
var.test(x, y)


sd(x)^2
sd(y)^2
sd(x)^2 / sd(y)^2