mydata = scan()

mydata

myts = ts(mydata, start = 1980)
plot(myts, ylab = "Labour Force Participation Data for age range 25-54")

## Possible models:
## ARIMA....Holt Linear Trend Method
## Model cannot exceed 100%
## Damping parameter of holt()
library(forecast)
library(ggplot2)
model_holt = holt(myts, h = 10)
plot(model_holt)

model_holt_damped = holt(myts, h = 10,damped = TRUE, phi = 0.8)
plot(model_holt_damped)
summary(model_holt_damped)


myarima = auto.arima(myts, stepwise = F, approximation = F)
myarima1 = forecast(myarima, h= 10) 
plot(myarima1)

autoplot(myts) +
  geom_line(size = 2)
  forecast::autolayer(model_holt$mean,
                      series = "Holt Linear Trend",size = 1.2) +
  forecast::autolayer(model_holt_damped$mean,
                      series = "Holt Linear Trend Damped",size = 1.2) +
  forecast::autolayer(myarima1$mean,series = "ARIMA",size = 1.2) +
  xlab("Year") +
  ylab("Labour Force Participation Data for age range 25-54") +
  guides(colour = guide_legend(title = "ForecastMethod")) +
  theme(legend.position = c(0.8,0.2)) +
  ggtitle("Singapore") +
  theme(plot.title = element_text(family = "Times",
                                  hjust = 0.5,
                                  color = "blue",
                                  face = "bold",
                                  size = 15))


  autoplot(myts) +
    geom_line(size = 2) +
  forecast::autolayer(model_holt$fitted,
                      series = "Holt Linear Trend",size = 1.2) +
    forecast::autolayer(model_holt_damped$fitted,
                        series = "Holt Linear Trend Damped",size = 1.2) +
    forecast::autolayer(myarima1$fitted,series = "ARIMA",size = 1.2) +
    xlab("Year") +
    ylab("Labour Force Participation Data for age range 25-54") +
    guides(colour = guide_legend(title = "ForecastMethod")) +
    theme(legend.position = c(0.8,0.2)) +
    ggtitle("Singapore") +
    theme(plot.title = element_text(family = "Times",
                                    hjust = 0.5,
                                    color = "blue",
                                    face = "bold",
                                    size = 15))
  