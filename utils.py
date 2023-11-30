import statsmodels.tsa.stattools as st
import numpy as np
import pandas as pd
import scipy.signal as scs
from sklearn import metrics

def adf_test(ts):
    """ Dickey-Fuller (DF) unit root test """
    result=st.adfuller(ts)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

class ConvertIntoStationary:
    """ Class to convert the TS into stationary using an additive model approach by model fitting and filtering """
    def __init__(self, order_trend=1, filter_freq=2.5, period = 12):
        self.order_trend = order_trend
        self.filter_freq = filter_freq
        self.period = period

    def convert_train(self, months, ts):
        """ Fit the components and converts the train TS into stationary """
        self.coefs = np.polyfit(months, np.array(ts), deg=self.order_trend)
        values = np.polyval(self.coefs, months)

        self.trend = pd.Series(values, index=ts.index, name='[Train]Trend Component')
        
        self.trend_adjusted = ts - self.trend

        self.sos = scs.butter(N=5, fs=self.period, Wn=[self.filter_freq], btype='lowpass', output='sos')
        seasonal = scs.sosfiltfilt(self.sos, self.trend_adjusted)
        self.seasonal_ts = pd.Series(data=seasonal, index=ts.index, name='[Train]Seasonal Component')

        self.trend_seasonal_adjusted = self.trend_adjusted - self.seasonal_ts
        self.trend_seasonal_adjusted.name = '[Train]Trend and seasonality adjusted'
        return self.trend_seasonal_adjusted
    
    def convert_test(self, months, ts):
        """ Transforms the test TS into stationary based on the results from the fit """
        values = np.polyval(self.coefs, months)

        self.trend_test = pd.Series(values, index=ts.index, name='[Test]Trend Component')
        
        self.trend_adjusted_test = ts - self.trend_test

        seasonal = scs.sosfiltfilt(self.sos, self.trend_adjusted_test)
        self.seasonal_ts_test = pd.Series(data=seasonal, index=ts.index, name='[Test]Seasonal Component')

        self.trend_seasonal_adjusted_test = self.trend_adjusted_test - self.seasonal_ts_test
        self.trend_seasonal_adjusted_test.name = '[Test]Trend and seasonality adjusted'
        return self.trend_seasonal_adjusted_test
    
def metrics_summary(ts_test, preds):
    print('MAPE:', metrics.mean_absolute_percentage_error(ts_test, preds))
    print('MAE:', metrics.mean_absolute_error(ts_test, preds))
    print('MSE:', metrics.mean_squared_error(ts_test, preds))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(ts_test, preds)))
    print('R2:', metrics.r2_score(ts_test, preds))


def reverse_diff(ts, preds, split_index, period):
    # reconstructs the test TS based on the train and predictions
    # the TS was converted into stationary using differencing of 1 and 12
    # uses the formulation
    #x'(n) = (x(n) - x(n-1)) - (x(n-12) - x(n-13))
    # <=> x'(n) = x(n) - x(n-1) - x(n-12) + x(n-13)
    # <=> x(n) = x'(n) + x(n-1) + x(n-12) - x(n-13)
    time_series = pd.concat((ts, preds))
    for i in range(len(ts), len(time_series)):
        time_series.iloc[i] = time_series.iloc[i] + time_series.iloc[i-1] + time_series.iloc[i-period] - time_series.iloc[i-(period+1)]
    return time_series[split_index:]


def get_predictions_from_horizon(model_fitted, forecast_horizon, ts, ts_test):
    """ Merge predictions according to a certain forecast horizon"""
    predictions = pd.Series()
    for i in range((len(ts_test) // forecast_horizon)+ 1):
        model_fitted = model_fitted.apply(pd.concat([ts, ts_test.iloc[:i*forecast_horizon]]))
        preds = model_fitted.forecast(steps=forecast_horizon)
        predictions = pd.concat([predictions, preds])
    predictions = predictions.iloc[:len(ts_test)]
    return predictions


def get_multi_predictions_from_horizon(fitted_model, train_data, test_data, explanatory_ts, ts, ts_test, forecast_horizon):
    """ Merge predictions according to a certain forecast horizon for multivariate data"""
    predictions = pd.Series()
    for i in range(len(ts_test) // forecast_horizon + 1):
        endog = pd.concat([ts, ts_test.iloc[:i*forecast_horizon]])
        exog = pd.concat([train_data[explanatory_ts], test_data.iloc[:i*forecast_horizon][explanatory_ts]])
        fitted_model = fitted_model.apply(endog=endog, exog=exog)
        aux = test_data[explanatory_ts].iloc[i*forecast_horizon:(i+1)*forecast_horizon]
        preds = fitted_model.forecast(steps=len(aux), exog=aux)
        predictions = pd.concat([predictions, preds])
    return predictions

def get_prediction_from_horizon_varmax(model_fitted, train_multi_data, test_multi_data, forecast_horizon):
    predictions = pd.DataFrame()
    for i in range(len(test_multi_data) // forecast_horizon + 1):
        model_fitted = model_fitted.apply(pd.concat([train_multi_data, test_multi_data.iloc[:i*forecast_horizon]]))
        preds = model_fitted.forecast(steps=forecast_horizon)
        predictions = pd.concat([predictions, preds])
    predictions = predictions.iloc[:len(test_multi_data)]
    return predictions