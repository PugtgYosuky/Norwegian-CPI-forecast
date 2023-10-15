import statsmodels.tsa.stattools as st
def adf_test(ts):
    """ Dickey-Fuller (DF) unit root test """
    result=st.adfuller(ts)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))