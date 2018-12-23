## To check seasonality 
## x <- ts(data, frequency=365/7)
# fit <- tbats(x)
# seasonal <- !is.null(fit$seasonal)

class(revenue$V2)

library(tidyr)
revenue = separate(revenue,
                   col = V2,
                   sep = c(2,-3),
                   into = c("rest","data","rest2"))
class(revenue$data)


myts = ts(as.numeric(revenue$data),
          start = 1997, frequency = 12)


summary(myts)

library(forecast)
myts = tsclean(myts) # tsclean() removes outliers and does missing value imputation by interpolation 

summary(myts)

plot(myts)

library(ggplot2)

mynetar = nnetar(myts)

## forecast 3 years with the model 
nnetforecast = forecast(mynetar, h = 36, PI = T)

autoplot(nnetforecast)

## interactive dygraph

data = nnetforecast$x
lower = nnetforecast$lower[,2]
upper = nnetforecast$upper[,2]
pforecast = nnetforecast$mean
mydata = cbind(data,lower,upper,pforecast)


library(dygraphs)
dygraph(mydata, main = "Oregon Composite Restaurant") %>%
  dyRangeSelector() %>%
  dySeries(name = "data", label = "Revenue Data") %>%
  dySeries(c("lower","pforecast","upper"), label = "Revenue Forecast") %>%
  dyLegend(show = "always", hideOnMouseOut = FALSE)%>%
  dyAxis("y", label = "Monthly Revenue USD") %>%
  dyHighlight(highlightCircleSize = 5, highlightSeriesOpts = list(strokeWidth = 2)) %>%
  dyOptions(axisLineColor = "navy", gridLineColor = "grey") %>%
  dyAnnotation("2010-08-01", text = "CF", tooltip = "Camp Festival", attachAtBottom = T)


