#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model of Seattle Home Prices Using Prophet
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

path = "/home/null/Documents/Python/Data/Real Estate/"
test_data = "HomePriceData.tsv"
working_test = "SeattleOverall.tsv"
extra_col = ['Unnamed: 0', 'Low (Under $403621)', 'Middle ($403621 - $625682)', 
             'High (Over $625682)']
fcst_len = 3

def process_data(dataframe):
    """
    Converts dataframe to format usable by Prophet.
    
    Parameters
        dataframe: dataframe of input data with date string and index values
        
    Returns
        dataframe: dataframe with two columns label ds (datetime values) and y
                   (index values).
    """
    # Convert date to datetime object.
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
    # Prophet expects two column labels: ds & y
    # ds column cannot be index.
    dataframe = dataframe.rename(columns={'DATE':'ds', 'Overall Market': 'y'})
    # Remove extra columns.
    dataframe = dataframe.drop(extra_col, axis=1)
    # Save a copy for inspection.
    dataframe.to_csv(path + working_test, sep='\t')
    return dataframe

def fuse_data(history, forecast):
    """
    Create a dateframe, indexed by date, containing actual and predicted values.
    
    Parameters
        history: dataframe of historical values
        forecast: dataframe of predicted values
        
    Return
        combined_data: dataframe cotaining historical and predicted values
    """
    combined_data = history.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']])
    combined_data = combined_data.set_index('ds')
    return combined_data

def visualize(data, model, forecast):
    """
    Create graphs of actual and predicted values, seasonality, and time series
    change points.
    
    Parameters
        data: dataframe containing historical and forecast values
        model: Prophet model
        forecast: dataframe with forecast
        
    Returns
        Nothing, outputs to console and files
    """
    # Set defaults
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 15, 15
    matplotlib.rcParams['savefig.format'] = 'png'
    matplotlib.rcParams['savefig.dpi'] = 'figure'
    # Create actual+forecast graph
    result_fig, result_ax = plt.subplots()
    labels = ['Actual', 'Forecast', 'Forecast Lower Bound', 'Forecast Upper Bound']
    result_ax.plot(data)
    result_ax.set_title('Seattle Home Prices January 1990 to May 2019', fontsize=18)
    result_ax.set_xlabel('Date')
    result_ax.set_ylabel('S&P Home Price Index')
    result_ax.legend(labels, loc='upper left')
    result_ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    result_fig.savefig(path + 'ForecastGraph')
    # Create seasonality graph
    components = model.plot_components(forecast)
    components.delaxes(components.get_axes()[0])
    components.get_axes()[0].set_title('Home Price Seasonality', fontsize=18)
    components.get_axes()[0].set_xlabel('Month')
    components.get_axes()[0].set_ylabel('Seasonality')
    components.set_size_inches((11,5), forward=True)
    components.savefig(path + 'ForecastSeasonality.png')
    # Create time series change point graph
    chg_pt = model.plot(forecast)
    add_changepoints_to_plot(chg_pt.gca(), model, forecast)
    chg_pt.get_axes()[0].set_title('Time Series Change Points', fontsize=18)
    chg_pt.get_axes()[0].set_xlabel('Date')
    chg_pt.get_axes()[0].set_ylabel('S&P Home Price Index')
    chg_pt.set_size_inches(13,12)
    chg_pt.savefig(path + 'ChangePointGraph')
    # Create loss table and graphs
    """
    Pandas has deprecated the 'Y' and 'M' values for timedelta. The following
    code will not work much longer:
    validation_data = cross_validation(model, horizon=pd.to_timedelta(3, unit='M'),
    period=pd.to_timedelta(1, unit='M'), initial=pd.to_timedelta(36, unit='M'))
    The lack of a timedelta monthly value that accounts for calendar variations
    makes modeling error of monthly series problematic.
    """
    validation_data = cross_validation(model, horizon='90 days', period='30 days', initial='365 days')
    loss_data = performance_metrics(validation_data)
    print(validation_data)
    print(loss_data.tail(n=5))
    mse_fig = plot_cross_validation_metric(validation_data, metric='mse')
    mse_fig.get_axes()[0].set_title('Mean Squared Error (MSE)', fontsize=18)
    mse_fig.savefig(path + 'MSEPlot')
    mape_fig = plot_cross_validation_metric(validation_data, metric='mape')
    mape_fig.get_axes()[0].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=18)
    mape_fig.savefig(path + 'MAPEPlot')

def main():
    with open(path + test_data, 'r') as f:
        data = pd.read_csv(f, sep='\t', header=0, index_col=False)
        # Format data.
        data = process_data(data)
        # Select training sample.
        training_sample = data[:-fcst_len]
        # Create the model and forecast values.
        model = Prophet(changepoint_range=0.98, changepoint_prior_scale=0.7)
 #       model = Prophet() Default settings with 80% range and 0.5 prior scale
        model.fit(training_sample)
        prediction_data = model.make_future_dataframe(periods=fcst_len, freq='M', include_history=True)
        forecast = model.predict(prediction_data)
        # Display results.
        plot_data = fuse_data(data, forecast)
        visualize(plot_data, model, forecast)

if __name__ == "__main__":
    # Workaround for Pandas plot bug. https://stackoverflow.com/a/57148021
    pd.plotting.register_matplotlib_converters()
    main()