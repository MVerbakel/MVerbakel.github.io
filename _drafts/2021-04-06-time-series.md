---
layout: post
title: "Time series analysis and modelling"
mathjax: true
---

Tips for analysis of time series data. 
Investigating changes
Method: reweight data to reflect earlier distribution of a feature. Does it explain the change?
Models.



Too often, I see forecasting either:
1. Overcomplicated: Applying complex ML models just to predict a moving average (?!), or
2. Oversimplified: Running regressions without understanding what the coefficients even mean.

I personally use 4 forecasting methods to model a range of outcomes, from conservative to aggressive:

1. ARIMA - Smooths time series data, w/o seasonality adjustment.
2. SARIMAX -  Like ARIMA, but accounts for seasonality. Likely to be the safest and conservative forecast.
3. Prophet -  Captures non-linear trends and seasonality. Often the most accurate. My favorite model for growth forecasts.
4. Manual Projection – aka Olga's secret, overly complicated manual projection. I plot every available metric’s historical D/D, W/W, M/M, and Y/Y % change and analyze their:
(a) correlations and relationships
(b) seasonal thresholds. 
It takes ages to complete, but it delivers the most precise forecast. 
If done right. If I can account for everything the teams are doing. Which is rarely the case. 😬 

When reporting, I typically present only Prophet alongside my Projection, keeping ARIMA and its variations for myself as checks.

There are many time series models out there: MA, AR, ARMA, ARIMA, SARIMA, Exponential Smoothing, VAR, and more. Forecasts are fun.