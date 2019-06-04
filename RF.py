from pandas import read_csv
from pandas import datetime
from pandas import concat
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


def parser(x):
    if len(x) == 8:
        x = '0' + x
    return datetime.strptime(x, '%d-%B-%y')


def table2lags(table, max_lag, min_lag=0, separator='_'):
    values = []
    for i in range(min_lag, max_lag + 1):
        values.append(table.shift(i).copy())
        values[-1].columns = [c + separator + str(i) for c in table.columns]
    return concat(values, axis=1)


series = read_csv('Kirie_Edited_Data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
lag = 6

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

series -= np.mean(series)
series /= np.std(series)
X = table2lags(series, lag-1)

name = 'P-TOT_0'
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
test_target = test[name]
train_target = train[name][lag:]
train = train[lag - 1:]

# Change the value here
rf = RandomForestRegressor(random_state=42)

# Model selection
param_grid = [{'n_estimators': [1000, 1500, 2000, 2500], 'max_features':[36, 18, 12]},
              {'bootstrap': [False], 'n_estimators':[100, 200, 300, 900], 'max_features':[36, 25, 20, 10]}]
grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)

grid_search.fit(train[:-1], train_target)
## Random oversampling
#from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_resample(train[:-1], train_target)
#grid_search.fit(X_resampled, y_resampled)

# rf.fit(train[:-1], train_target)

best_rf = grid_search.best_estimator_

print(best_rf)

predictions = best_rf.predict(test)

error = mean_absolute_error(test_target, predictions)
error_2 = mean_squared_error(test_target, predictions)
print('Test MSE: %.3f' % error_2)
print('Test MAE: %.3f' % error)
# plot
pyplot.plot(test_target.data)
pyplot.plot(predictions, color='red')
pyplot.show()
