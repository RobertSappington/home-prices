Home Price Forecasting Using Facebook’s Prophet
=====================================

This project uses Facebook’s Prophet to create a benchmark model of home prices for use in evaluating performance of other home price models. I chose Prophet to test the package’s functionality and performance. I’m using data from the S&P CoreLogic Case-Shiller Seattle Home Price NSA Index. Note:This project forecasts price trends in a residential real estate market, not individual home prices.

Real Estate Data Considerations
---------------------------------------

Real estate prices provide a challenging data set for forecasting because the data is both cyclical and seasonal and affected by many, often conflicting, causal factors. While forecasts during stable trend phases can be useful, accurate forecasts of inflection points offer the highest rewards. The current time series (from January 1990 to May 2019) provides an interesting window for forecasting because data from June 2018 to May 2019 suggest, but do not confirm, a change in the cyclical phase. This lateral index movement has occurred twice in prior periods: once before a growth phase (1991) and once in the middle of a decline phase (2010). At the time of this writing, the flat-lining index coincides with conflicting indicators: full employment and low interest rates support growth while an inverted yield curve and volatile capital markets signal decline. Periods like this provide an interesting forecast challenge.

Prophet Model
------------------

For the first iteration, I created a forecast without modifying Prophet’s defaults. The graph of the default model forecast looks like classic overfitting—a very tight fit with historical training data but divergence from test data.

![](https://github.com/RobertSappington/home-prices/blob/master/plots/ForecastGraphDefaultAnnotated.png)

Reading the Prophet docs reveals the modeling logic responsible for the result:

“Prophet detects changepoints by first specifying a large number of potential changepoints at which the rate is allowed to change. It then puts a sparse prior on the magnitudes of the rate changes (equivalent to L1 regularization) - this essentially means that Prophet has a large number of possible places where the rate can change, but will use as few of them as possible . . . By default changepoints are only inferred for the first 80% of the time series in order to have plenty of runway for projecting the trend forward and to avoid overfitting fluctuations at the end of the time series.”

The changepoints mapped in the default settings clearly exclude the end of series change.

![](https://github.com/RobertSappington/home-prices/blob/master/plots/ChangePointGraphDefaultAnnotated.png)

While the default logic is sound in the abstract, this situation requires a rigorous evaluation of end-of-series phenomenon. I tweaked the `changepoint_range` parameter to include the June 2018 changepoint. See graph below.

![](https://github.com/RobertSappington/home-prices/blob/master/plots/ChangePointGraphTunedAnnotated.png)

This change, combined with minor tuning of the `changepiont_prior_scale` parameter, produced a forecast that better reflects the end-of-series inflection.

![](https://github.com/RobertSappington/home-prices/blob/master/plots/ForecastGraphTunedAnnotated.png)

Conclusions
---------------

With only minor changes to settings and no additional regressors, Prophet produced a time-series forecast that handled trend, seasonality, and residuals well. This tool enables forecasters to focus on higher-order bits, e.g. modeling end-of-series perturbations to enhance model robustness.

Environment
----------------
- Python 3.7
- Prophet 0.5

Data Source
---------------
[S&P CoreLogic Case-Shiller Seattle Home Price NSA Index](https://us.spindices.com/indices/real-estate/sp-corelogic-case-shiller-seattle-home-price-nsa-index)
