library(DBI)
library(odbc)
con <- dbConnect(odbc::odbc(), Driver = "Oracle in instantclient_23_8",
		DBQ = "OracleDB", UID = "EDU119", PWD = "oracle_4U")
df2 <- dbGetQuery(con, "SELECT * FROM KDT_SEPSIS_CRF_SOFA")
head(print(df2))


## apache2
con <- dbConnect(odbc::odbc(), Driver = "Oracle in instantclient_23_8",
		DBQ = "OracleDB", UID = "EDU119", PWD = "oracle_4U")

df2 <- dbGetQuery(con, "SELECT * FROM KDT_SEPSIS_CRF_APACHE")
head(df2)

df <- dbGetQuery(con, 'SELECT * FROM KDT_SEPSIS_CRF_APACHE WHERE "Age" > 65')
head(df)
str(df)

df3 <- dbGetQuery(con, "SELECT * FROM KDT_SEPSIS_CRF_SOFA")
head(df3)


#### 로컬에서 #####
library(readxl)
test <- read_excel("C:/Users/KDT_09/hjkim01_git/clinical data analysis by oracle DB/kdt_sepsis_crf_sofa.xlsx")
str(test)