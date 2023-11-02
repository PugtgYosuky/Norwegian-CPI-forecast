import statsmodels.tsa.stattools as st
import numpy as np
import pandas as pd
import scipy.signal as scs

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