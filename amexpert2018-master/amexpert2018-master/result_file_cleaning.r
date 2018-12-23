#Wroking with external files
getwd() # to know the default working directory or folder for R
setwd("D:/Analytics_Vidhya_Research/AM_Expert_2018 Competition")
getwd()

Data1 <- read.csv("submission_r.csv")
Data2 <- read.csv("test.csv")

library(sqldf)
Data3 <- sqldf("SELECT DISTINCT a.session_id ,count(1) FROM Data1 a  group by session_id having count(1)>1")

Data4 <- sqldf("SELECT DISTINCT a.session_id ,is_click FROM Data1 a
               where a.session_id in (select session_id from Data3) and is_click = 1 order by session_id, is_click")

Data5 <- sqldf("SELECT DISTINCT a.session_id ,is_click FROM Data1 a
               where a.session_id in (select session_id from Data3) and is_click = 1 
               union all
               SELECT DISTINCT a.session_id ,is_click FROM Data1 a
               where a.session_id not in (select session_id from Data3) ")
write.csv(Data5, file = "MyData.csv")
