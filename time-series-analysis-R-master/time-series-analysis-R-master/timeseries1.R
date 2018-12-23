## https://cran.r-project.org/web/views/TimeSeries.html

plot(lynx); length(lynx)
plot(LakeHuron);length(LakeHuron)
plot(nottem);length(nottem)
plot(AirPassengers);length(AirPassengers)
plot(EuStockMarkets);length(EuStockMarkets)
plot(sunspot.year);length(sunspot.year)
?rnorm


# Here you can find all relevant links to this course:
#   
#   1. R task view on time series analysis:
#   
#   https://cran.r-project.org/web/views/TimeSeries.ht...
# 
# 2. Package lubridate overview:
#   
#   https://cran.r-project.org/web/packages/lubridate/...
# 
# 3. Article on time and date classes in R (starting page 30):
#   
#   https://cran.r-project.org/doc/Rnews/Rnews_2004-1....
# 
# 4. Mini Book on statistical forecasting:
#   
#   http://people.duke.edu/~rnau/411home.htm
# 
# 5. Online Course on time series with R examples:
#   
#   https://onlinecourses.science.psu.edu/stat510/node...
# 
# 6. Free Ebook on time series analysis by R Hyndman
# 
# https://otexts.org/fpp2/


x = as.POSIXct("2015-12-25 11:45:34")
y = as.POSIXlt("2015-12-25 11:45:34")
unclass(x)
unclass(y)
y$zone

x
y
x= as.Date("2015-12-25")
x
class(x)
unclass(x)

?unclass
library(chron)
x = chron("12/25/2015", "23:34:09")
x
class(x)
unclass(x)


library(lubridate)

## different ways in how to input dates
ymd(20180906)
dmy(24091984)
mdy(9241984)

## lets use time and date together
mytimepoint <- ymd_hm("20180924 11:24", tz = "Europe/Prague")
mytimepoint

## extracting the component of it
minute(mytimepoint)
day(mytimepoint)
hour(mytimepoint)


## we can also change values within our object
hour(mytimepoint) <- 15
mytimepoint
class(mytimepoint)

## which time zones do we have available ....time zone recognition also depends on your location and machine 
olson_time_zones() ## deprecated
OlsonNames()


#lets check which day our time point is 
wday(mytimepoint)


## label to display the name of the day .... no abbreviation
wday(mytimepoint, label = T, abbr = F)

## we can calculate which time  our timepoint would be in another time zone
with_tz(mytimepoint, tz = "Asia/Kolkata")


## time intervals
time1 = ymd_hm("1993-09-23 11:23", tz = "Europe/Prague")
time2 = ymd_hm("1995-11-22 11:23", tz = "Europe/Prague")

## getting the interval 
myinterval = interval (time1, time2); myinterval
class(myinterval) ## interval is an object class from lubridate


## Excercise
a = ymd(c(19980101, 19990101, 20000101, 20010101), tz = "CET")
b = hms(c("22 4 5","4-9-45", "11:9:56","23 23 59"))
f = rnorm(4,10) ; f = round(f, digits = 2) ; f
#rnorm(<number of random samples>, <mean of those samples>)


datetimemeasurement = cbind.data.frame(date = a, time = b , mesaure = f)
datetimemeasurement


## calculations with time
minutes(7)
## errors
minutes(2.5)
## get the duration of minutes
dminutes(3)
## how to add minutes and seconds
minutes(2) + seconds(67)
# class duration to perform additions like above
as.duration(minutes(2) + seconds(67))


## lubridate has an array  of time  classes, period or duration differ


## which year was a leap year 
leap_year(2009:2014)

ymd(20140101) + years(1)
ymd(20140201) + dyears(1)

## lets do whole thing with a leap year
leap_year(2016)
ymd(20160101) + years(1)
ymd(20160201) + dyears(1)

## dyears - duration years will always take into 365 days for a year  but with "years" will increase the year unit by +1

?runif

## Creating Time Series
mydata = runif(n= 50, min = 10, max = 45)
mydata

## ts for class time series 
## Data starts in 1956 - 4 observations / year (quarterly)
mytimeseries = ts(data = mydata,
                  start = c(1956,3), frequency = 4)

plot(mytimeseries)
class(mytimeseries)
time(mytimeseries)

## Write notes from notebook here

## hourly measurements with daily patterns starts at 8AM on the first day
## start = c(1,8), frequency = 24

## Measurement taken twice a day on weekdays with weekly patterns, starts at the first week
## start = 1, frequency = 10 ## NA for holidays - regular spacing 

## Monthly measurements with yearly cycle - frequency = 12

## Weekly measurements with yearly cycle - frequency = 52

## While specifying the start and frequency arguments 
## ... think about the cycle and the number of measurements per cycle

## the above method applies with mts class too

?cumsum

##task

set.seed(123)
x = cumsum(rnorm(n=450))
x
gr_ts = ts(data = x, start = c(1914,11), frequency = 12)
plot(gr_ts)


## added from solution
library(lattice)
xyplot.ts(gr_ts)

## standard rbase plots
plot(nottem)

## plot of components
plot(decompose(nottem))

library(forecast)
library(ggplot2)

## Directly plotting a forecast of a model
plot(forecast(auto.arima(nottem)), h = 5)

## if the data is not time series then we can use plot.ts(vector)

## ggplot equivalent to plot
autoplot((nottem))

## ggplots work with different layers
autoplot(nottem) + ggtitle("Autoplot of Nottingham temp data")


## Time series specific plots
ggseasonplot(nottem)

ggmonthplot(nottem)

class(nottem)

View(AirPassengers)
class(AirPassengers)

seasonplot(AirPassengers,
           year.labels = TRUE,
           col = c("red","blue"),
           xlab = "",
           labelgap = 0.35,
           type = "l",
           bty = "l",
           cex = 0.75
           )
           
?seasonplot


## Very important code section
library(zoo)
library(tidyr)

?separate

irreg.split = 
  separate(
    irregular_sensor,
    col = X1,
    into = c("date","time"),
    sep = 8,
    remove = T
  )

View(irreg.split)

sensor.date = strptime(irreg.split$date,'%m/%d/%y')
sensor.date

## Creating a data frame 
irregts.df = data.frame(
  date = as.Date(sensor.date),
                 measurement = irregular_sensor$X2)

View(irregts.df)


##Getting a zoo object
irreg.dates = zoo(irregts.df$measurement,
                  order.by = irregts.df$date)
irreg.dates


## regularizing with aggregate
ag.irregtime = aggregate(irreg.dates, as.Date, mean)
ag.irregtime


##########
## Method 2 - date and time component kept
sensor.date1 = strptime(irregular_sensor$X1,'%m/%d/%y %I:%M %p')
sensor.date1





##Getting a zoo object
irreg.dates1 = zoo(irregular_sensor$X2,
                  order.by = sensor.date1)
irreg.dates1

plot(irreg.dates1)

## regularizing with aggregate
ag.irregtime = aggregate(irreg.dates1, as.Date, sum)
ag.irregtime
plot(ag.irregtime)


## Handling outliers and missing data 
myts = ts(ag.irregtime)
plot(myts)

## convert the 2nd column to a simple ts 
myts = ts(ts_NAandOutliers$mydata)
myts

summary(myts)
plot(myts)

myts1 = tsoutliers(myts)
myts1
class(myts1)
plot(myts)

## last observation carried forward 
myts.NAlocf = na.locf(myts)

## values provided beforehand
myts.NAfill = na.fill(myts,33)

## na.trim trims the NA values at the borders

##interp = interpolation ... formula for interpolation is .... (y-y1)(x2-x1) = (y2-y1)(x-x1)
myts.NAinterp = na.interp(myts)
mytsclean = tsclean(myts) ## tslean() = tsoutliers() + tsinterp()
plot(mytsclean)
summary(mytsclean)


## xts


## Three Simple Methods
## Naive method - last observation carried forward method
## ... projects the last observation into the future
## use the naive() function in the forecast package
## the function can be tweaked to even fit a seasonal dataset
## example - to forecast feb18 r takes the last observed value of feb17
## snaive()


## Average method
## Calculates the mean of the data and projects that into the future
## use the meanf() function from the forecast package

## Drift method
## Calculate the diference between first and last observation and carries that increase into the future
## use the rwf() function from the forecast package



## example dataset 
set.seed(95)
myts = ts(rnorm(200), start=(1818))
plot(myts)

library(forecast)
meanm = meanf(myts, h = 20)
naivem = naive(myts,h = 20)
driftm = rwf(myts, h = 20, drift = T)

plot(meanm, PI= F, main="")

lines(naivem$mean, col=123, lwd = 2)

lines(driftm$mean, col = 22, lwd = 2)

legend("topleft",lty = 1, col=c(4,123,22),
       legend= c("Mean Method","Naive method","Drift method"))

## Model comparison and accuracy 

## MAE - Mean Absolute Error
## The mean of all differences between actual and forecasted absolute values
## MAE = takes absolute values 

## RMSE - Root Mean Squared Error/Deviation to offset signages
## The sample standard deviation of differences between actual and forecasted values

# MASE - Mean Absolute Scaled Error
# Measures the forecast error compared to the error of a naive forecast
# 0 < x < 1
# x = 1 as good as naive forecast - always picking the last value observed
# x = 0.5 the model has double the prediction accuracy as a naive last value approach
## which means the lower the value of x the better
# x> 1 means the model needs a lot of improvement

# MAPE - Mean Absolute Percentage Error
# Measures the difference of forecast errors and divides it by the actual observation value
# Does not allow for 0 values
# Puts much more weight on extreme values and positive errors
# scale independent - you can use it to compare a model on different datasets

# AIC - Akaike Information Criterion 
# Common measure in forecasting, statistical modeling and machine learning
# it is great to compare the complexity of different models
# Penalizes more complex models
# the lower the AIC score the better

set.seed(95)
myts = ts(rnorm(200),start= 1818)
mytstrain = window(myts,start = 1818, end = 1988)
plot(mytstrain)

library(forecast)
meanm = meanf(mytstrain, h = 30)
naivem = naive(mytstrain, h = 30)
driftm = rwf(mytstrain, h = 30, drift = T)


mytstest = window(myts, start = 1988)

accuracy(meanm, mytstest) ## we get the lowest values of almost all main paramters discussed above so it is the best
accuracy(naivem, mytstest)
accuracy(driftm, mytstest)

## Rule of thumb - We want all the patterns in the model, only randomness should stay in the residuals ....general quality check of the models
## Residuals should be the container of randomness (data that cannot be explained in mathematical terms)
## .... ideally they have a mean of zero and constant variance
## The residuals should be uncorrelated (correlated residuals still have information left in them)
## .... ideally they are normally distributed
## A non zero mean can be easily fixed with addition or subtraction, while correlations can be extracted via 
## ... modeling tools (e.g. differencing) - ensuring normal distribution (constant variance)
## ... might be possible in some cases , however, transformations (e.g. logarithms) might help

var(meanm$residuals)
mean(meanm$residuals)

mean(naivem$residuals)

naivewithoutNA = naivem$residuals
naivewithoutNA = naivewithoutNA[2:170]
is.na(naivewithoutNA)
var(naivewithoutNA)
mean(naivewithoutNA)


hist(naivem$residuals)
acf(naivewithoutNA)


## in acf plot - several bars range outside the threshold meaning the the data has significant autocorrelation
## 4/20 bars are over / below the thresholds - the residuals still have information left in them
## Improving the model (e.g. applying a transformation) might reduce the bars


## STationarity Test 
## has the same statistical properties thoughout the time series
## statistical properties - variance, mean, autocorrelation
## most analytical procedures in time series require stationary data 
## if the data lacks stationarity there are transformations to be applied to make the data stationary  or it can be changed via differencing
## Differencing adjusts the data according to the time spans that differ in e.g. variance or mean (extensively used in ARIMA models)


## De-Trending
## loads of time series have a trend in it --> mean changes as a result of of the trend 
## ... causes underestimated predictions
## Solution:
## 1. Test if you get staionarity if you de-trend the data set : take the trend component out of the dataset --> trend stationarity
## 2. If the above procedure is not enough then we can use differencing --> difference stationarity
## 3. Unit-root tests tell whether there is a trend stationarity or a difference stationarity 
###### the first difference goes from one period to the very next one (two successive steps)
###### The first difference is stationarity and random --> random walk (each value is a random step away from a previous value)
###### The first difference is stationary but not completely random (e.g. values are autocorrelated) --> require a more sophisticated model (e.g. exponential smoothing, ARIMA)
## urca package - unit root tests 
## library tseries - adf.test(x) - the Augmented Dickey Fuller Test removes the autocorrelation and tests for non - stationarity
## funitroot tests package - for mnore advanced tests
x = rnorm(1000)
library(tseries)
adf.test(x)
plot(x)

plot(nottem)
plot(decompose(nottem))
adf.test(nottem)


y = diffinv(x)
plot(y)
adf.test(y)




## Autocorrelation
### It is a statistical term which describes the correlation (or the lack of such) in a time series dataset
### It is a key statistic, because it tells you whether previous observations influence the recent one -> correlation on a time scale
### Lags: Steps on a time scale
### For best statistical results, you always need to find out whether autocorrelation is present 
### There are many tools available in R to test for autocorrelation, but in most cases it is clear to see whether it is present - this comes from the functional knowledge
### for example there won't be any autocorrelation in a random walk, while the lynx dataset has it for sure

## acf() - Autocorrelation function between different timelags in a time series .... it returns a measure
## pacf() - Partial Autocorrelation function 
## Durbin-Watson Test - library(lmtest)
## Gets the autocorrelation only of the first order - between one time point and the immediate successor
## its not robust so use it with caution

## in hypothesis testing - if p value is < 0.05 then we accept the Alternate hypothesis

## acf() and pacf() functions
## Functions acf() and pacf() make sense on time series data
## Alternatively we can try all possible models one by one --> time consuming
## Using these functions provides a sytematic way to identify the paramters
## Autocorrelation : the correlation coefficient between different time points (lags) in a time series
## Partial Autocorrelation: The correlation coefficient adjusted for all shorter lags in a time series
## the acf() is used to identify the moving average (MA) part of the ARIMA model, while pacf() identifies the values for the autoregressive part (AR)
## both functions are part of rBase

####
acf(lynx, lag.max = 20, plot = F)
## reg acf function above 
## Several bars ranging out of the 95% confidence intervals
## Omit the first bar - it is the autocorrelation against itself at lag0
## the first 2 lags are significant 


####
pacf(lynx, lag.max = 20, plot = F)
## reg pacf function above 
## PACF starts at lag1
## The first lag is a significant lag , the second lag is significant to the negatiove side

## for normal distribution autocorrelation and partial auto correlation would be negligent as it is random 
acf(rnorm(500),lag.max = 20)
## only 1 bar is a little above 95% confidence interval which is fine as that is permitted outside our level of significance

## 1 = absolute positive correlation
## -1 = absolute negative correlation


## tsdisplay 
tsdisplay(rnorm(500), lag.max = 20)

## Practice excercise
?rnorm

set.seed(54)
myts = ts(c(rnorm(50,34,10),
          rnorm(67,7,1),
          runif(23,3,14)))

myts = log(myts) ## when data is positive then logarithm

plot(myts)

## mean and variance are difference

library(forecast)
meanm = meanf(myts, h= 10)
naivem = naive(myts, h = 10)
driftm = rwf(myts, h = 10, drift = T)

plot(meanm, main= "", bty = "l")
lines(naivem$mean, col = 123, lwd = 2)
lines(driftm$mean, col = 22, lwd = 2)
legend("bottomleft",lty = 1, col= c(4,123,22),
       bty = "n", cex = 0.75,
       legend = c("Mean Method","Naive Method","Drift Method"))

length(myts)

mytrain = window(myts , start = 1, end = 112)
mytest = window(myts, start = 113)

meanma = meanf(mytrain, h= 28)
naivema = naive(mytrain, h = 28)
driftma = rwf(mytrain, h = 28, drift = T)

accuracy(meanma, mytest)
accuracy(naivema, mytest)
accuracy(driftma, mytest)



plot(naivem$residuals)
mean(naivem$residuals[2:140])
hist(naivem$residuals)
shapiro.test(naivem$residuals)
acf(naivem$residuals[2:140])

tsdisplay(myts)



#### Selecting a Suitable Model

## Qualitative 
## Quantitaive - Linear and Non Linear
## Linear - Simple Models, Exponential Smoothing, ARIMA, Seasonal Decomposition
## Non Linear Models - Neural Nets, Support Vector Machines, Clustering 


## Simple - Last Observation carried forward model 
## Drift model
## Mean model
## USe them to model random data with no pattern
## Bench for other models 

## Exponential smoothing  - Trend and seasonality are key determinants 
## Can put more weight on recent observations

## ARIMA Model - Explains patterns in the data based on autoregression

## Seasonal Decomposition - Datsets needs to be seasonal or at least have a frequency 
## Minimum number of seasonal cycles (2)
## Recent variations of seasonal decomposition
## SEATS 
## X11
## STL Decomposition

## Further Linear Models 
## Linear Regressions
## Dynamic Rgressions
## Vector Autoregressive Models (library: vars) [Multivariate Time Series Analysis]


## Non-Linear Models 
## Neural Nets - 
## Tries to model the brain's neuron system 
## An input vector is compressed to several layers
## Each layer consists of multiple neurons
## Weight of importance may be ascribed to each neuron
## The amount of required layer is specified by the dataset
## library forecast - nnetar()
## library nnfor
## SVM but not used extensively 
## Clustering  - kml package 

## Seasonal Decomposition
## Univariate Seasonal Time Series 
## Modelling options
## Seasonal ARIMA 
## Holt-Winters Exponential Smoothing 
## Seasonal Decomposition
## To perform seasonal decomposition, the dataset must have a seasonal component 
## Frequency parameter for generated data 
## Frequently measured data: inflation rates, weather measurements, etc
## Seasonal decomposition decomposes seasonal time series data to its components 
## Trend
## Seasonality 
## Remainder - random data
## Additive method - Adds component up ... Constant seasonal component .... When seasonal component remains constant then this method can be opted
## Multiplicative Method - Multiplies component 


## Drawbacks of seasonal decomposition
## NA Values
## Slow to catch sudden changes
## Constant seasonality 

## Alternatives - SEATS, x11, stl decomposition
## Values for all observation - non NA
## Seasonal part can be adjusted over time 
## Tools - decompose(), stl() ... forecast integration stl generated objects , stlf()
## library seasonal: seas()

library(ggplot2)
## Decomposing time series (U)

plot(nottem)
frequency(nottem)
length(nottem)
decompose(nottem, type = "additive")
plot(decompose(nottem, type = "additive"))
autoplot(decompose(nottem, type = "additive"))
### Signs when additive model can be used - constant amplitude and no trend

## Alternatively the function stl could be used
plot(stl(nottem, s.window = "periodic"))


## Decomposition Demo 


## Seasonal adjustments 
mynottem = decompose(nottem, "additive")
class(mynottem)

## We are subtracting the seasonal element 
nottemadjusted = nottem - mynottem$seasonal


plot(nottemadjusted)
plot(mynottem$seasonal)
plot(nottem)

## a stl forecast from teh package forecast 
library(forecast)
plot(stlf(nottem, method = "arima"))

## Seasonal Decomposition Excercise 
plot(AirPassengers)
frequency(AirPassengers)


## Airpassengers
## Trend
## Seasonal pattern
## Increasing Amplitude
## Multiplicative Model

mymodel1 = decompose(AirPassengers, type = "additive")
mymodel2 = decompose(AirPassengers, type = "multiplicative")


plot(mymodel1)
plot(mymodel2)

## Dataset without the seasonal plot
plot(mymodel1$trend + mymodel1$random)

## Multiplicative seasonality and additive trend

## Exponential smoothing with ets



## Simple Moving Average
## Smoothing the data 
## Getting the dataset closer to the center by evening out the highs 
## ... and the lows -> decreasing the impact of extreme values
## Classic smoother : Simple Moving Average  - Widely used in Science and finance (trading)
## How does SMA work - 
## Define the number of observations to use and take thier average 
## Period - Successive values of a time series
## Works best with non-seasonal data
## Ideal for getting the general trend remving white noise


## SMOOTHING
library(TTR)

## in order to identify trends, we can use smoothers
## ... like a simple moving average
## ... n identifies the order or the SMA - one can experiment with this parameter

x = c(1,2,3,4,5,6,7)
SMA(x, n = 3)

lynx_smoothed = SMA(lynx, n = 6)
lynx_smoothed

## can compare the smoothed vs the original lynx data
## only applicable for non seasonal datasets
plot(lynx)
plot(lynx_smoothed)





## Exponential Smoothing with ETS
## Describe the time series with three paramters
## Error - additive, multiplicative (x>0)
## Trend - non-present, additive, multiplicative
## Seasonality - non-present, additive, multiplicative

## Values are either summed up, multiplied or omitted (in case of zeroes or negatives as it might not reflect correctly)

## Parameters can be mixed - e.g. additive trend with multiplicative seasonality : Multiplicative Holt-Winters model

## Exponential smoothing : recent data is given more weight than the older observations

## R functions
## Simple Exponential Smoothing - ses() - for datasets without trend and seasonality
## Holt linear exponential smoothing model - holt() - for datasets with a trend and without seasonality - argument 
## ... 'damped' to damp down the trend over time 
## Holt-Winters seasonal exponential smoothing - hw() : for data with trend 
## ... and seasonal component + damping parameter 
## Above models are set manually 
## Automated model selection via ets() [library forecast()]
## Model selection based on information criteria but models can be customized


## For three parameters errors, trend and seasonality smoothing coefficients are 
## ... there to manage the weighting based on the timestamp
## reactive model relies heavily on recent data - high coefficient ~1 
## smooth model - low coefficient ~ 0
## Coefficients 
## alpha : initial level
## beta : trend
## gamma - seasonality 
## phi - damped paramter 
## Required argument for ets() : data
## Argument 'model' for pre-selecting a model 
## Default 'ZZZ' .... autoselection of the three components
## - Additive 'A' .... Multiplicative 'M' .... non-present 'N'
## Coefficients and boundaries can also be pre-set

## ets
library(forecast)

## using function ets 
etsmodel = ets(nottem)
etsmodel

## plotting the model vs original 
plot(nottem, lwd=3)
lines(etsmodel$fitted, col = "red")

## plotting the forecast
plot(forecast(etsmodel, h = 12))

## changing the prediction interval 
plot(forecast(etsmodel, h = 12, level = 95))


## Manually setting the ets model to multiplicative model
etsmodmult = ets(nottem, model = "MZM")

## Not a good model as all coefficient and info values are inflated as compared to the automated model
etsmodmult

## plot as comparison
plot(nottem, lwd = 3)
lines(etsmodmult$fitted, col = "red")

## Judgemental or Qualititative forecast
## Delphi method
## 5-20 forecasters (professionals)
## anonymous setting

## Forecasting by analogy e.g. Real Estate

## Sceanrio Based Forecasting e.g. Black Swan extreme scenario


##### ARIMA - Very popular univariate time series 

## Introduction to ARIMA Models

## UnivaRIATE, non-seasonal ARIMA models

## ARIMA(p,d,q)

plot(lynx)
library(forecast)

tsdisplay(lynx)

auto.arima(lynx)

auto.arima(lynx, trace = T,
           stepwise = F,
           approximation = F)


x = c(1,2,3,4,5,6,7)

SMA(x, n = 3)


## ARIMA models 
## ARIMA calculations
## AR(2) model
myarima = arima(lynx, order = c(2,0,0))
myarima


tail(lynx)
residuals(myarima)

##  ARIMA Simulators
set.seed(123)
asim = arima.sim(model = list ( order = c(1,0,1),
                                ar = c(0.4),
                                ma = c(0.3)),
                                n = 1000) +10 ## 10 is the mean of the samples take n for simulation
plot(asim)

library(zoo)
plot(rollmean(asim,50))
plot(rollmean(asim,25))

library(tseries)
adf.test(asim) ## if p value less than 0.05 then alternative hypothesis is true

library(forecast)
tsdisplay(asim)

?auto.arima
auto.arima(asim, trace = T, stepwise = F, approximation = F)


## Manual ARIMA Parameter Selection
## test for stationarity 
adf.test(lynx)

tsdisplay(lynx)

myarima = Arima(lynx, order = c(2,0,0))
tsdisplay(lynx)
checkresiduals(myarima)


myarima = Arima(lynx, order = c(4,0,0))
checkresiduals(myarima)


## Simulation 
set.seed(123)
myts = arima.sim(model = list (order = c(0,0,2),
                               ma = c(0.3,0.7)),
                 n = 1000) + 10

adf.test(myts)

tsdisplay(myts)

myarima = Arima(myts, order = c(0,0,3))
checkresiduals(myarima) ## p value in its output indicates that whether the residulas are normally distributed or not

auto.arima(myts)

## ARIMA Forecasting
myarima = auto.arima(lynx, stepwise = F, approximation = F)
arimafore = forecast(myarima, h = 10)
plot(arimafore)

arimafore$mean
plot(arimafore, xlim = c(1930,1944))


myets = ets(lynx)
etsfore = forecast(myets,h = 10)

#Comparison plot for 2 models
library(ggplot2)
autoplot(lynx) +
  forecast::autolayer(etsfore$mean,series = 'ETS Model')+
  forecast::autolayer(arimafore$mean,series = 'ARIMA Model')+
  xlab('year') +
  ylab('Lynx Trappings') +
  guides(
    colour = guide_legend(
      title = 'Forcast Method'
    )
  ) +
  theme(
    legend.position = c(0.8,0.8)
  )


## ARIMA with explanatory variables

library(ggplot2)
ggplot(cyprinidae,
       aes(y = concentration, x = X1)) + 
  geom_point() +
  aes(colour = predator_presence)

x = ts(cyprinidae$concentration)
y = cyprinidae$predator_presence

mymodel = auto.arima(x , xreg = y,
                     stepwise = F,
                     approximation = F)
mymodel
checkresiduals(mymodel)

y1 = c(T,T,F,F,F,F,T,F,T,F)
 
plot(forecast(mymodel, xreg = y1))
plot(forecast(mymodel, xreg = y1),
     xlim = c(230,260))



## 3rd project 
library(xts)
library(zoo)
library(quantmod)

novartis = getSymbols("NVS",
                      auto.assign = F,
                      from = "2015-01-01",
                      to = "2016-01-01")


plot(as.ts(novartis$NVS.Open))

chartSeries(novartis, type = "line")
library(forecast)
ggtsdisplay(novartis$NVS.Open)

novartisarima = auto.arima(novartis$NVS.Open,
                      stepwise = T,
                      approximation = F,
                      trace = T)

novartisarima2 = Arima(novartis$NVS.Open, order = c(1,1,1))
novartisarima2



## forecast arima
plot(forecast(novartisarima, h = 20))

plot(forecast(novartisarima2, h = 20))

## ETS modeln)
plot(forecast(novartisets, h = 20))

novartis_df = as.data.frame(novartis)
novartis_df$df_date = rownames(novartis_df)
novartis_df$df_date = as.Date(novartis_df$df_date)

                                                         
head(novartis_df)

mydates = seq.Date(from = as.Date("2015-01-01"),
                   to = as.Date("2016-01-01"),
                   by = 1)
mydates = data.frame(df_date= mydates)
mydates$df_date = mydates$Date

mydata = merge(novartis_df,mydates, by = 'df_date',all.y = TRUE)


mydata = mydata[5:366,]
mydata = mydata[-seq(from = 7, to = nrow(mydata), by = 7),]
mydata = mydata[-seq(from = 6, to = nrow(mydata), by = 6),]
mydata = na.locf(mydata)

highestprice = ts(as.numeric(mydata$NVS.High),
                  frequency = 5)
seasonplot(highestprice, season.labels = c("Mon","Tue","Wed","Thu","Fri"))
monthplot(highestprice)
monthplot(highestprice,base = median, col.base = "red")

par(mfrow=c(1,2))
lowestprice = ts(as.numeric(mydata$NVS.Low), frequency = 5)
monthplot(lowestprice,base = median, col.base = "red")
monthplot(highestprice,base = median, col.base = "red")
par(mfrow=c(1,1))

plot(stl(highestprice, s.window = "periodic"))
