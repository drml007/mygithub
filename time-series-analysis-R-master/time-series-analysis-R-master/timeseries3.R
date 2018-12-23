## use frequency in ts function in the code for seasonality

## Seasonal Decomposition

mydata = scan()
plot.ts(mydata)

germaninfl = ts(mydata, start = 2008, frequency = 12)
 plot (germaninfl)
 
plot( decompose(germaninfl))

plot(stl(germaninfl, s.window = 7)) 

library(forecast)

plot(stlf(germaninfl, method = "ets"))
plot(forecast(ets(germaninfl), h = 24))

library(ggplot2)
autoplot(stlf(germaninfl, method = "ets"))

## Seasonal Arima (package forecast)
germaninflarima = auto.arima(germaninfl,stepwise = T, approximation = F, trace = T)

forec = forecast(germaninflarima)
plot(forec)


germaninflets = ets(germaninfl)
plot(forecast(germaninflets, h = 60))
plot(hw(germaninfl, h = 60)) ## automatically genrates the forecast

## cross validation of the two models
forecastets = function(x,h){
  forecast(ets(x),h=h)
}

forecastarima = function(x,h){
  forecast(auto.arima(x),stepwise = T, approximation = F, trace = T, h = h)
}


etserror = tsCV(germaninfl, forecastets, h = 1)
arimaerror = tsCV(germaninfl, forecastarima, h = 1)


mean(etserror^2, na.rm = TRUE)
mean(arimaerror^2, na.rm = TRUE)









