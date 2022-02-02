library(tidyverse)
library(forecast)
library(lubridate)
library(fpp3)

#1.Import Data
df<-read_csv('train_all.csv')
df<-df%>%select(-'X1')
df<-df%>%group_by(Store,Date,IsHoliday)%>%summarise('Weekly_Sales'=sum(Weekly_Sales),
                                                    'Temperature'=mean(Temperature),
                                                    'Fuel_Price'=mean(Fuel_Price),
                                                    'CPI'=mean(CPI),
                                                    'Unemployment'=mean(Unemployment),
                                                    'Size'=mean(Size))
df%>%group_by(Store)%>%count()%>%filter(n!=143)                                

#2. Split into validation and train data set
df$Date<-as.Date(df$Date)
val<-df%>%filter(Date>='2012-03-31')
val%>%group_by(Store)%>%count()%>%filter(n!=31)

train<-df%>%filter(Date<'2012-03-31')
train%>%group_by(Store)%>%count()%>%filter(n!=112)
################################################################
#Trial Modeling on Store1
#3-1. Trial on Store 1
tem_train<-train%>%filter(Store==2)
tem_test<-val%>%filter(Store==2)
tem_train<-tem_train[,-1]
tem_test<-tem_test[,-1]

#3-2. Rename y variable as y
tem_train<-tem_train%>%rename("y"=Weekly_Sales)
tem_test<-tem_test%>%rename('y'=Weekly_Sales)

#3-3. Transform y into time series
tem_train$y<-ts(tem_train$y,
                frequency = 365.25/7,
                start = decimal_date(ymd(min(tem_train$Date))))

tem_test$y<-ts(tem_test$y,
               frequency = 365.25/7,
               start = decimal_date(ymd(min(tem_test$Date))))

#3-4. Getting the Regressor
train_reg<-as.matrix(tem_train[,c(2,4,5,6,7)])
test_reg<-as.matrix(tem_test[,c(2,4,5,6,7)])

#3-5. Create the model
set.seed(999)
nnetar_model <- nnetar(y = tem_train$y,
                       decay = 0,
                       size = 10,
                       xreg = train_reg)

#3-6. Forecast
forecast = forecast(nnetar_model, xreg = test_reg)
forecast$mean
result=data.frame('Store'=2,
                  'prediction'=as.numeric(forecast$mean))
str(result)
plot(forecast)

#3-7. Accuracy
accuracy(forecast$mean, tem_test$y)

################################################################
#Loop for the 45 stores
set.seed(999)
for (i in 1:45) {
  #Step1
  tem_train<-train%>%filter(Store==i)
  tem_test<-val%>%filter(Store==i)
  tem_train<-tem_train[,-1]
  tem_test<-tem_test[,-1]
  
  #Step2: Rename y variable
  tem_train<-tem_train%>%rename("y"=Weekly_Sales)
  tem_test<-tem_test%>%rename('y'=Weekly_Sales)
  
  #Step3: Transform y into timeseries variable
  tem_train$y<-ts(tem_train$y,
                  frequency = 365.25/7,
                  start = decimal_date(ymd(min(tem_train$Date))))
  
  tem_test$y<-ts(tem_test$y,
                 frequency = 365.25/7,
                 start = decimal_date(ymd(min(tem_test$Date))))
  #Step4: Extract External Regressors
  train_reg<-as.matrix(tem_train[,c(2,4,5,6,7)])
  test_reg<-as.matrix(tem_test[,c(2,4,5,6,7)])
  
  #Step5: Modeling
  nnetar_model <- nnetar(y = tem_train$y,
                         decay = 0.1,
                         size = 10,
                         xreg = train_reg)
  
  #Step6: Result
  forecast = forecast(nnetar_model, xreg = test_reg)
  pred=data.frame('Store'=i,
                  'Date'=tem_test$Date,
                  'prediction'=as.numeric(forecast$mean))
  
  #Step7
  if(i==1){
    result<-pred
  }else{
    result<-result%>%rbind(pred)
  }
}
  
val<-val%>%arrange(Store,Date)
accuracy(result$prediction,val$Weekly_Sales)
