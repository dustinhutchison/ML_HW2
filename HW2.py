#From Vincent Tatan's In 12 minutes:Stocks Analysis with Pandas and Scikit-Learn
#Homework2 for Siraj ML class - none of this code is mine, only modified and stock pick changed
import datetime
from pandas_datareader import data as pdr
import yfinance as yf

#pick stock, VTI
VTI = yf.Ticker("VTI")

#import data
start = datetime.datetime(2016, 1, 1)
end = datetime.datetime.now()

yf.pdr_override()

df = pdr.get_data_yahoo("VTI", start=start, end=end)


dfreg = df.loc[:,['Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

import math
import numpy as np
from sklearn import preprocessing, model_selection


dfreg.fillna(value=-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(dfreg)))


forecast_col = 'Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))


X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
#print('Dimension of X',X.shape)
#print('Dimension of y',y.shape)

# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#build models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso

# Linear regression
clfreg = LinearRegression (n_jobs=-1)
clfreg.fit (X_train, y_train)

# Lasso
clfl = Lasso ()
clfl.fit (X_train, y_train)

# Ridge
clfr = Ridge ()
clfr.fit (X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor (n_neighbors=2)
clfknn.fit (X_train, y_train)

#test model
confidencereg = clfreg.score(X_test, y_test)
confidencel = clfl.score(X_test, y_test)
confidencer = clfr.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)

print("The linear regression confidence is ",confidencereg)
print("The lasso confidence is ",confidencel)
print("The ridge 3 confidence is ",confidencer)
print("The knn regression confidence is ",confidenceknn)


def plotPrediction(_forecast_set, _predictionName, _lastDate):
    #plot prediction
    dfreg['Forecast'] = np.nan

    last_unix = _lastDate
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in _forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

    #%matplotlib inline
    import matplotlib.pyplot as plt
    from matplotlib import style

    # Adjusting the size of matplotlib
    import matplotlib as mpl
    mpl.rc('figure', figsize=(8, 7))

    # Adjusting the style of matplotlib
    style.use('ggplot')

    dfreg['Close'].tail(300).plot()
    dfreg['Forecast'].tail(300).plot()

    plt.title(_predictionName)
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.show()


# Printing the all forecasts, I did change the spelling of Prediction and added the stock symbol
last_date = dfreg.iloc[-1].name

forecast_set = clfreg.predict(X_lately)
print("VTI linear regression prediction: ")
print(forecast_set)
plotPrediction(forecast_set, "VTI linear regression prediction", last_date)

forecast_set = clfl.predict(X_lately)
print("VTI lasso regression prediction: ")
print(forecast_set)
plotPrediction(forecast_set, "VTI lasso prediction", last_date)

forecast_set = clfr.predict(X_lately)
print("VTI ridge regression prediction: ")
print(forecast_set)
plotPrediction(forecast_set, "VTI ridge prediction", last_date)

forecast_set = clfknn.predict(X_lately)
print("VTI knn regression prediction: ")
print(forecast_set)
plotPrediction(forecast_set, "VTI knn regression prediction", last_date)
