library(sjmisc)
library(tidyverse)
library(tidyverse)
library(forecast)
library(lubridate)
library(fpp3)
#1. Import Data
test<-read_csv('test.csv')
test<-test%>%arrange(Store,Date)
features<-read_csv('features.csv')
stores<-read_csv('stores.csv')
train<-read_csv('train_all.csv')

#2.Tidy whole train data
train<-train%>%select(-'X1')
train<-train%>%group_by(Store,Date,IsHoliday)%>%summarise('Weekly_Sales'=sum(Weekly_Sales),
                                                          'Temperature'=mean(Temperature),
                                                          'Fuel_Price'=mean(Fuel_Price),
                                                          'CPI'=mean(CPI),
                                                          'Unemployment'=mean(Unemployment),
                                                          'Size'=mean(Size))
#3. Predict CPI & Unemployment
for (i in 1:45) {
  tem<-features%>%filter(Store==i)
  tem<-tem%>%arrange(Date)
  index=which(is.na(tem$CPI))
  for(j in index){
    tem$CPI[j]<-mean(tem$CPI[(j-5):(j-1)])
    tem$Unemployment[j]<-mean(tem$Unemployment[(j-5):(j-1)])
  }
  if(i==1){
    features_final<-tem
  }else{
    features_final<-features_final%>%rbind(tem)
  }
}

#4.Tidy Test Data
test<-test%>%left_join(select(features_final,-IsHoliday),by=c("Store",'Date'))
test<-test%>%left_join(select(stores,-Type),by='Store')
test<-test%>%group_by(Store,Date,IsHoliday)%>%summarise('Temperature'=mean(Temperature),
                                                        'Fuel_Price'=mean(Fuel_Price),
                                                        'CPI'=mean(CPI),
                                                        'Unemployment'=mean(Unemployment),
                                                        'Size'=mean(Size))
test%>%group_by(Store)%>%count()%>%filter(n!=39)

#5. Forecast on Test Data
#Loop for the 45 stores
set.seed(999)
for (i in 1:45) {
  #Step1
  tem_train<-train%>%filter(Store==i)
  tem_test<-test%>%filter(Store==i)
  tem_train<-tem_train[,-1]
  tem_test<-tem_test[,-1]
  
  #Step2
  tem_train<-tem_train%>%rename("y"=Weekly_Sales)
  
  #Step3
  tem_train$y<-ts(tem_train$y,
                  frequency = 365.25/7,
                  start = decimal_date(ymd(min(tem_train$Date))))
  
  #Step4
  train_reg<-as.matrix(tem_train[,c(2,4,5,6,7)])
  test_reg<-as.matrix(tem_test[,c(2,3,4,5,6)])
  
  #Step5
  nnetar_model <- nnetar(y = tem_train$y,
                         decay = 0.1,
                         size = 10,
                         xreg = train_reg)
  
  #Step6
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

#6. Save Result
write_csv(result,'final_prediction.csv')
